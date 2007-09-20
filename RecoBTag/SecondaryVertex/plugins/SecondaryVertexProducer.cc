#include <functional>
#include <algorithm>
#include <iterator>
#include <cstddef>
#include <limits>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <boost/iterator/transform_iterator.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"
#include "RecoBTag/SecondaryVertex/interface/VertexFilter.h"

// #define DEBUG

using namespace reco;

class SecondaryVertexProducer : public edm::EDProducer {
    public:
	explicit SecondaryVertexProducer(const edm::ParameterSet &params);
	~SecondaryVertexProducer();

	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	const edm::InputTag		trackIPTagInfoLabel;
	TrackIPTagInfo::SortCriteria	sortCriterium;
	TrackSelector			trackSelector;
	edm::ParameterSet		vtxRecoPSet;
	bool				withPVError;
	VertexFilter			vertexFilter;
	bool				vertex3dSorting;
};

static TrackIPTagInfo::SortCriteria getSortCriterium(const std::string &name)
{
	if (name == "sip3dSig")
		return TrackIPTagInfo::IP3DSig;
	if (name == "prob3d")
		return TrackIPTagInfo::Prob3D;
	if (name == "sip2dSig")
		return TrackIPTagInfo::IP2DSig;
	if (name == "prob2d")
		return TrackIPTagInfo::Prob2D;
	if (name == "sip2dVal")
		return TrackIPTagInfo::IP2DValue;

	throw cms::Exception("InvalidArgument")
		<< "Identifier \"" << name << "\" does not represent a valid "
		<< "track sorting criterium." << std::endl;
}

SecondaryVertexProducer::SecondaryVertexProducer(
					const edm::ParameterSet &params) :
	trackIPTagInfoLabel(params.getParameter<edm::InputTag>("trackIPTagInfos")),
	sortCriterium(getSortCriterium(params.getParameter<std::string>("trackSort"))),
	trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
	vtxRecoPSet(params.getParameter<edm::ParameterSet>("vertexReco")),
	withPVError(params.getParameter<bool>("usePVError")),
	vertexFilter(params.getParameter<edm::ParameterSet>("vertexCuts")),
	vertex3dSorting(params.getParameter<bool>("vertex3dSorting"))
{
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
	edm::ESHandle<TransientTrackBuilder> trackBuilder;
	es.get<TransientTrackRecord>().get("TransientTrackBuilder",
	                                   trackBuilder);

	edm::Handle<TrackIPTagInfoCollection> trackIPTagInfos;
	event.getByLabel(trackIPTagInfoLabel, trackIPTagInfos);

	// result secondary vertices

	std::auto_ptr<SecondaryVertexTagInfoCollection>
			tagInfos(new SecondaryVertexTagInfoCollection);

	// use beamspot as fallback (FIXME: use same as TrackIPTagInfos)

	BeamSpot beamSpot;
	Vertex beamSpotPV(beamSpot.position(), beamSpot.covariance3D(),
	                  -1, -1, 0);

#ifdef DEBUG
	std::cout << "Found " << trackIPTagInfos->size() << " trackIPTagInfos" << std::endl;
#endif

	for(TrackIPTagInfoCollection::const_iterator iterJets =
		trackIPTagInfos->begin(); iterJets != trackIPTagInfos->end();
		++iterJets) {

		std::vector<SecondaryVertexTagInfo::IndexedTrackData> trackData;

		const Vertex &pv = iterJets->primaryVertex().isNonnull()
					? *iterJets->primaryVertex()
					: beamSpotPV;

		edm::RefToBase<Jet> jetRef = iterJets->jet();

		GlobalVector jetDir(jetRef->momentum().x(),
		                    jetRef->momentum().y(),
		                    jetRef->momentum().z());

		std::vector<std::size_t> indices =
				iterJets->sortedIndexes(sortCriterium);

		const TrackRefVector &trackRefs =
					iterJets->sortedTracks(indices);

		const std::vector<TrackIPTagInfo::TrackIPData> &ipData =
					iterJets->impactParameterData();

#ifdef DEBUG
		std::cout << "Has " << indices.size() << " tracks." << std::endl;
#endif

		// build transient tracks

		std::vector<TransientTrack> tracks;
		for(unsigned int i = 0; i < indices.size(); i++) {
			typedef SecondaryVertexTagInfo::IndexedTrackData IndexedTrackData;

			const TrackRef &trackRef = trackRefs[i];

			trackData.push_back(IndexedTrackData());
			tracks.push_back(trackBuilder->build(trackRef));

			trackData.back().first = i;

			// select tracks for SV fit

			trackData.back().second.svStatus =
				trackSelector(*trackRef, ipData[i])
					? SecondaryVertexTagInfo::TrackData::trackUsedForVertexFit
					: SecondaryVertexTagInfo::TrackData::trackSelected;
		}
#ifdef DEBUG
		std::cout << "Have " << tracks.size() << " selected tracks." << std::endl;
#endif

		// try to fit vertex

		std::vector<SecondaryVertex> SVs;
		try {
			ConfigurableVertexReconstructor vertexReco(vtxRecoPSet);

			// give fitter the selected tracks

			std::vector<reco::TransientTrack> fitTracks;
			for(unsigned int i = 0; i < trackData.size(); i++)
				if (trackData[i].second.usedForVertexFit())
					fitTracks.push_back(tracks[i]);

#ifdef DEBUG
			std::cout << "Having " << fitTracks.size() << " selected tracks." << std::endl;
#endif

			// perform fit

			std::vector<TransientVertex> fittedSVs;
			fittedSVs = vertexReco.vertices(fitTracks);
#ifdef DEBUG
			std::cout << "Found " << fittedSVs.size() << " vertices." << std::endl;
#endif

			// build combined SV information and filter

			SVBuilder svBuilder(pv, jetDir, withPVError);
			std::remove_copy_if(boost::make_transform_iterator(
			                    	fittedSVs.begin(), svBuilder),
			                    boost::make_transform_iterator(
			                    	fittedSVs.end(), svBuilder),
			                    std::back_inserter(SVs),
			                    SVFilter(vertexFilter,
			                             pv, jetDir));
		} catch(...) {
			// most likely the following problem:
			// fewer than two significant tracks (w > 0.001)
			// note that if this catch is removed,
			// CMSSW can fail on valid events, sorry pals...
		}

		// identify most probable SV (closest to interaction point)
		// FIXME: identify if this is the best strategy!

		const SecondaryVertex *bestSV = 0;
		double bestValue = std::numeric_limits<double>::max();

		for(std::vector<SecondaryVertex>::const_iterator iter =
			SVs.begin(); iter != SVs.end(); iter++) {

#ifdef DEBUG
		std::cout << "dist3d = (" << iter->dist3d().value() << ", " << iter->dist3d().error() << ") -> " << iter->dist3d().significance() << std::endl;
		std::cout << "dist2d = (" << iter->dist2d().value() << ", " << iter->dist2d().error() << ") -> " << iter->dist2d().significance() << std::endl;
#endif
			double value = std::abs(vertex3dSorting
					? iter->dist3d().significance()
					: iter->dist2d().significance());
			if (value < bestValue) {
#ifdef DEBUG
			std::cout << "... is new best SV" << std::endl;
#endif
				bestValue = value;
				bestSV = &*iter;
			}
		}

		if (!bestSV)
			continue;

		// mark tracks successfully used in vertex fit

		for(track_iterator iter = bestSV->tracks_begin();
		    iter != bestSV->tracks_end(); iter++) {
			TrackRefVector::const_iterator pos =
				std::find(trackRefs.begin(), trackRefs.end(),
				          *iter);
			if (pos == trackRefs.end())
				throw cms::Exception("TrackNotFound")
					<< "Could not find track in secondary "
					   "vertex in origina tracks."
					<< std::endl;

			unsigned int index = pos - trackRefs.begin();
			trackData[index].second.svStatus = 
				SecondaryVertexTagInfo::TrackData::trackAssociatedToVertex;
		}

		// fill result into tag infos

#ifdef DEBUG
		std::cout << "saving as tag info" << std::endl;
#endif
		tagInfos->push_back(
			SecondaryVertexTagInfo(
				trackData, *bestSV,
				bestSV->dist2d(), bestSV->dist3d(),
				SVs.size(), jetDir,
				TrackIPTagInfoRef(trackIPTagInfos,
					iterJets - trackIPTagInfos->begin())));
	}

	event.put(tagInfos);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SecondaryVertexProducer);
