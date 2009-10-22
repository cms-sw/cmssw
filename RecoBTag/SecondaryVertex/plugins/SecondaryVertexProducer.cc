#include <functional>
#include <algorithm>
#include <iterator>
#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <boost/iterator/transform_iterator.hpp>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/TrackSorting.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/VertexFilter.h"
#include "RecoBTag/SecondaryVertex/interface/VertexSorting.h"

using namespace reco;

namespace {
	template<typename T>
	struct RefToBaseLess : public std::binary_function<edm::RefToBase<T>,
							   edm::RefToBase<T>,
							   bool> {
		inline bool operator()(const edm::RefToBase<T> &r1,
				       const edm::RefToBase<T> &r2) const
		{
			return r1.id() < r2.id() ||
			       (r1.id() == r2.id() && r1.key() < r2.key());
		}
	};
}

class SecondaryVertexProducer : public edm::EDProducer {
    public:
	explicit SecondaryVertexProducer(const edm::ParameterSet &params);
	~SecondaryVertexProducer();

	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	enum ConstraintType {
		CONSTRAINT_NONE	= 0,
		CONSTRAINT_BEAMSPOT,
		CONSTRAINT_PV_BEAMSPOT_SIZE,
		CONSTRAINT_PV_BS_Z_ERRORS_SCALED,
		CONSTRAINT_PV_ERROR_SCALED,
		CONSTRAINT_PV_PRIMARIES_IN_FIT
	};

	static ConstraintType getConstraintType(const std::string &name);

	const edm::InputTag		trackIPTagInfoLabel;
	TrackIPTagInfo::SortCriteria	sortCriterium;
	TrackSelector			trackSelector;
	ConstraintType			constraint;
	double				constraintScaling;
	edm::ParameterSet		vtxRecoPSet;
	bool				withPVError;
	double				minTrackWeight;
	VertexFilter			vertexFilter;
	VertexSorting			vertexSorting;
};

SecondaryVertexProducer::ConstraintType
SecondaryVertexProducer::getConstraintType(const std::string &name)
{
	if (name == "None")
		return CONSTRAINT_NONE;
	else if (name == "BeamSpot")
		return CONSTRAINT_BEAMSPOT;
	else if (name == "BeamSpot+PVPosition")
		return CONSTRAINT_PV_BEAMSPOT_SIZE;
	else if (name == "BeamSpotZ+PVErrorScaledXY")
		return CONSTRAINT_PV_BS_Z_ERRORS_SCALED;
	else if (name == "PVErrorScaled")
		return CONSTRAINT_PV_ERROR_SCALED;
	else if (name == "BeamSpot+PVTracksInFit")
		return CONSTRAINT_PV_PRIMARIES_IN_FIT;
	else
		throw cms::Exception("InvalidArgument")
			<< "SecondaryVertexProducer: ``constraint'' parameter "
			   "value \"" << name << "\" not understood."
			<< std::endl;
}

SecondaryVertexProducer::SecondaryVertexProducer(
					const edm::ParameterSet &params) :
	trackIPTagInfoLabel(params.getParameter<edm::InputTag>("trackIPTagInfos")),
	sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
	trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
	constraint(getConstraintType(params.getParameter<std::string>("constraint"))),
	constraintScaling(1.0),
	vtxRecoPSet(params.getParameter<edm::ParameterSet>("vertexReco")),
	withPVError(params.getParameter<bool>("usePVError")),
	minTrackWeight(params.getParameter<double>("minimumTrackWeight")),
	vertexFilter(params.getParameter<edm::ParameterSet>("vertexCuts")),
	vertexSorting(params.getParameter<edm::ParameterSet>("vertexSelection"))
{
	if (constraint == CONSTRAINT_PV_ERROR_SCALED ||
	    constraint == CONSTRAINT_PV_BS_Z_ERRORS_SCALED)
		constraintScaling = params.getParameter<double>("pvErrorScaling");

	produces<SecondaryVertexTagInfoCollection>();
}

SecondaryVertexProducer::~SecondaryVertexProducer()
{
}

namespace {
	struct SVBuilder :
		public std::unary_function<const Vertex&, SecondaryVertex> {

		SVBuilder(const reco::Vertex &pv,
		          const GlobalVector &direction,
		          bool withPVError) :
			pv(pv), direction(direction),
			withPVError(withPVError) {}

		SecondaryVertex operator () (const reco::Vertex &sv) const
		{ return SecondaryVertex(pv, sv, direction, withPVError); }

		const Vertex		&pv;
		const GlobalVector	&direction;
		bool			withPVError;
	};

	struct SVFilter :
		public std::unary_function<const SecondaryVertex&, bool> {

		SVFilter(const VertexFilter &filter, const Vertex &pv,
		         const GlobalVector &direction) :
			filter(filter), pv(pv), direction(direction) {}

		inline bool operator () (const SecondaryVertex &sv) const
		{ return !filter(pv, sv, direction); }

		const VertexFilter	&filter;
		const Vertex		&pv;
		const GlobalVector	&direction;
	};
			
} // anonynmous namespace

void SecondaryVertexProducer::produce(edm::Event &event,
                                      const edm::EventSetup &es)
{
	typedef std::map<TrackBaseRef, TransientTrack,
	                 RefToBaseLess<Track> > TransientTrackMap;

	edm::ESHandle<TransientTrackBuilder> trackBuilder;
	es.get<TransientTrackRecord>().get("TransientTrackBuilder",
	                                   trackBuilder);

	edm::Handle<TrackIPTagInfoCollection> trackIPTagInfos;
	event.getByLabel(trackIPTagInfoLabel, trackIPTagInfos);

	edm::Handle<BeamSpot> beamSpot;
	unsigned int bsCovSrc[7] = { 0, };
	double sigmaZ = 0.0, beamWidth = 0.0;
	switch(constraint) {
	    case CONSTRAINT_PV_BEAMSPOT_SIZE:
		event.getByType(beamSpot);
		bsCovSrc[3] = bsCovSrc[4] = bsCovSrc[5] = bsCovSrc[6] = 1;
		sigmaZ = beamSpot->sigmaZ();
		beamWidth = beamSpot->BeamWidthX();
		break;

	    case CONSTRAINT_PV_BS_Z_ERRORS_SCALED:
		event.getByType(beamSpot);
		bsCovSrc[0] = bsCovSrc[1] = 2;
		bsCovSrc[3] = bsCovSrc[4] = bsCovSrc[5] = 1;
		sigmaZ = beamSpot->sigmaZ();
		break;

	    case CONSTRAINT_PV_ERROR_SCALED:
		bsCovSrc[0] = bsCovSrc[1] = bsCovSrc[2] = 2;
		break;

	    case CONSTRAINT_BEAMSPOT:
	    case CONSTRAINT_PV_PRIMARIES_IN_FIT:
		event.getByType(beamSpot);
		break;

	    default:
		/* nothing */;
	}

	ConfigurableVertexReconstructor vertexReco(vtxRecoPSet);

	TransientTrackMap primariesMap;

	// result secondary vertices

	std::auto_ptr<SecondaryVertexTagInfoCollection>
			tagInfos(new SecondaryVertexTagInfoCollection);

	for(TrackIPTagInfoCollection::const_iterator iterJets =
		trackIPTagInfos->begin(); iterJets != trackIPTagInfos->end();
		++iterJets) {

		std::vector<SecondaryVertexTagInfo::IndexedTrackData> trackData;

		const Vertex &pv = *iterJets->primaryVertex();

		std::set<TransientTrack> primaries;
		if (constraint == CONSTRAINT_PV_PRIMARIES_IN_FIT) {
			for(Vertex::trackRef_iterator iter = pv.tracks_begin();
			    iter != pv.tracks_end(); ++iter) {
				TransientTrackMap::iterator pos =
					primariesMap.lower_bound(*iter);

				if (pos != primariesMap.end() &&
				    pos->first == *iter)
					primaries.insert(pos->second);
				else {
					TransientTrack track =
						trackBuilder->build(
							iter->castTo<TrackRef>());
					primariesMap.insert(pos,
						std::make_pair(*iter, track));
					primaries.insert(track);
				}
			}
		}

		edm::RefToBase<Jet> jetRef = iterJets->jet();

		GlobalVector jetDir(jetRef->momentum().x(),
		                    jetRef->momentum().y(),
		                    jetRef->momentum().z());

		std::vector<std::size_t> indices =
				iterJets->sortedIndexes(sortCriterium);

		TrackRefVector trackRefs = iterJets->sortedTracks(indices);

		const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
					iterJets->impactParameterData();

		// build transient tracks used for vertex reconstruction

		std::vector<TransientTrack> fitTracks;
		for(unsigned int i = 0; i < indices.size(); i++) {
			typedef SecondaryVertexTagInfo::IndexedTrackData IndexedTrackData;

			const TrackRef &trackRef = trackRefs[i];

			trackData.push_back(IndexedTrackData());
			trackData.back().first = indices[i];

			// select tracks for SV finder

			if (!trackSelector(*trackRef, ipData[i], *jetRef,
			                   RecoVertex::convertPos(
			                   		pv.position()))) {
				trackData.back().second.svStatus =
					SecondaryVertexTagInfo::TrackData::trackSelected;
				continue;
			}

			TransientTrackMap::const_iterator pos =
					primariesMap.find(
						TrackBaseRef(trackRef));
			if (pos != primariesMap.end()) {
				primaries.erase(pos->second);
				fitTracks.push_back(pos->second);
			} else
				fitTracks.push_back(
					trackBuilder->build(trackRef));

			trackData.back().second.svStatus =
				SecondaryVertexTagInfo::TrackData::trackUsedForVertexFit;
		}

		// perform actual vertex finding

		std::vector<TransientVertex> fittedSVs;
		switch(constraint) {
		    case CONSTRAINT_NONE:
			fittedSVs = vertexReco.vertices(fitTracks);
			break;

		    case CONSTRAINT_BEAMSPOT:
			fittedSVs = vertexReco.vertices(fitTracks, *beamSpot);
			break;

		    case CONSTRAINT_PV_BEAMSPOT_SIZE:
		    case CONSTRAINT_PV_BS_Z_ERRORS_SCALED:
		    case CONSTRAINT_PV_ERROR_SCALED: {
			BeamSpot::CovarianceMatrix cov;
			for(unsigned int i = 0; i < 7; i++) {
				unsigned int covSrc = bsCovSrc[i];
				for(unsigned int j = 0; j < 7; j++) {
					double v;
					if (!covSrc || bsCovSrc[j] != covSrc)
						v = 0.0;
					else if (covSrc == 1)
						v = beamSpot->covariance(i, j);
					else
						v = pv.covariance(i, j) *
						    constraintScaling;
					cov(i, j) = v;
				}
			}

			BeamSpot bs(pv.position(), sigmaZ,
			            beamSpot->dxdz(), beamSpot->dydz(),
			            beamWidth, cov, BeamSpot::Unknown);

			fittedSVs = vertexReco.vertices(fitTracks, bs);
		    }	break;

		    case CONSTRAINT_PV_PRIMARIES_IN_FIT: {
			std::vector<TransientTrack> primaries_(
					primaries.begin(), primaries.end());
			fittedSVs = vertexReco.vertices(primaries_, fitTracks,
			                                *beamSpot);
		    }	break;
		}

		// build combined SV information and filter

		std::vector<SecondaryVertex> SVs;
		SVBuilder svBuilder(pv, jetDir, withPVError);
		std::remove_copy_if(boost::make_transform_iterator(
		                    	fittedSVs.begin(), svBuilder),
		                    boost::make_transform_iterator(
		                    	fittedSVs.end(), svBuilder),
		                    std::back_inserter(SVs),
		                    SVFilter(vertexFilter, pv, jetDir));

		// clean up now unneeded collections

		fitTracks.clear();
		fittedSVs.clear();

		// sort SVs by importance

		std::vector<unsigned int> vtxIndices = vertexSorting(SVs);

		std::vector<SecondaryVertexTagInfo::VertexData> svData;

		svData.resize(vtxIndices.size());
		for(unsigned int idx = 0; idx < vtxIndices.size(); idx++) {
			const SecondaryVertex &sv = SVs[vtxIndices[idx]];

			svData[idx].vertex = sv;
			svData[idx].dist2d = sv.dist2d();
			svData[idx].dist3d = sv.dist3d();
			svData[idx].direction =
				GlobalVector(sv.x() - pv.x(),
				             sv.y() - pv.y(),
				             sv.z() - pv.z());

			// mark tracks successfully used in vertex fit

			for(Vertex::trackRef_iterator iter = sv.tracks_begin();
			    iter != sv.tracks_end(); iter++) {
				if (sv.trackWeight(*iter) < minTrackWeight)
					continue;

				TrackRefVector::const_iterator pos =
					std::find(trackRefs.begin(), trackRefs.end(),
					          iter->castTo<TrackRef>());
				if (pos == trackRefs.end())
					throw cms::Exception("TrackNotFound")
						<< "Could not find track from secondary "
						   "vertex in original tracks."
						<< std::endl;

				unsigned int index = pos - trackRefs.begin();
				trackData[index].second.svStatus =
					(SecondaryVertexTagInfo::TrackData::Status)
					((unsigned int)SecondaryVertexTagInfo::TrackData::trackAssociatedToVertex + idx);
			}
		}

		// fill result into tag infos

		tagInfos->push_back(
			SecondaryVertexTagInfo(
				trackData, svData, SVs.size(),
				TrackIPTagInfoRef(trackIPTagInfos,
					iterJets - trackIPTagInfos->begin())));
	}

	event.put(tagInfos);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SecondaryVertexProducer);
