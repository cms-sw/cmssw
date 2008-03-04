#include <functional>
#include <algorithm>
#include <iterator>
#include <cstddef>
#include <string>
#include <vector>

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
	double				minTrackWeight;
	VertexFilter			vertexFilter;
	VertexSorting			vertexSorting;
};

SecondaryVertexProducer::SecondaryVertexProducer(
					const edm::ParameterSet &params) :
	trackIPTagInfoLabel(params.getParameter<edm::InputTag>("trackIPTagInfos")),
	sortCriterium(TrackSorting::getCriterium(params.getParameter<std::string>("trackSort"))),
	trackSelector(params.getParameter<edm::ParameterSet>("trackSelection")),
	vtxRecoPSet(params.getParameter<edm::ParameterSet>("vertexReco")),
	withPVError(params.getParameter<bool>("usePVError")),
	minTrackWeight(params.getParameter<double>("minimumTrackWeight")),
	vertexFilter(params.getParameter<edm::ParameterSet>("vertexCuts")),
	vertexSorting(params.getParameter<edm::ParameterSet>("vertexSelection"))
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

	for(TrackIPTagInfoCollection::const_iterator iterJets =
		trackIPTagInfos->begin(); iterJets != trackIPTagInfos->end();
		++iterJets) {

		std::vector<SecondaryVertexTagInfo::IndexedTrackData> trackData;

		const Vertex &pv = *iterJets->primaryVertex();

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

		// build transient tracks

		std::vector<TransientTrack> tracks;
		for(unsigned int i = 0; i < indices.size(); i++) {
			typedef SecondaryVertexTagInfo::IndexedTrackData IndexedTrackData;

			const TrackRef &trackRef = trackRefs[i];

			trackData.push_back(IndexedTrackData());
			tracks.push_back(trackBuilder->build(trackRef));

			trackData.back().first = indices[i];

			// select tracks for SV fit

			trackData.back().second.svStatus =
				trackSelector(*trackRef, ipData[i], *jetRef)
					? SecondaryVertexTagInfo::TrackData::trackUsedForVertexFit
					: SecondaryVertexTagInfo::TrackData::trackSelected;
		}

		// try to fit vertex

		std::vector<SecondaryVertex> SVs;
		try {
			ConfigurableVertexReconstructor vertexReco(vtxRecoPSet);

			// give fitter the selected tracks

			std::vector<reco::TransientTrack> fitTracks;
			for(unsigned int i = 0; i < trackData.size(); i++)
				if (trackData[i].second.usedForVertexFit())
					fitTracks.push_back(tracks[i]);

			// perform fit

			std::vector<TransientVertex> fittedSVs;
			fittedSVs = vertexReco.vertices(fitTracks);

			// build combined SV information and filter

			SVBuilder svBuilder(pv, jetDir, withPVError);
			std::remove_copy_if(boost::make_transform_iterator(
			                    	fittedSVs.begin(), svBuilder),
			                    boost::make_transform_iterator(
			                    	fittedSVs.end(), svBuilder),
			                    std::back_inserter(SVs),
			                    SVFilter(vertexFilter,
			                             pv, jetDir));
		} catch(VertexException e) {
			// most likely the following problem:
			// fewer than two significant tracks (w > 0.001)
			// note that if this catch is removed,
			// CMSSW can fail on valid events...
			// validation can check if the the TagInfo collection
			// contains anything at all
		}

		// identify most probable SV

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
