#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"

//the selectores used to select the tracks
#include "Alignment/CommonAlignmentProducer/interface/AlignmentTracksFromVertexSelector.h"

// the following include is necessary to clone all track branches
// including recoTrackExtras and TrackingRecHitsOwned.
// if you remove it the code will compile, but the cloned
// tracks have only the recoTracks branch!
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"

struct TrackFromVertexConfigSelector {
  typedef std::vector<const reco::Track *> container;
  typedef container::const_iterator const_iterator;
  typedef reco::TrackCollection collection;

  TrackFromVertexConfigSelector(const edm::ParameterSet &cfg, edm::ConsumesCollector &&iC) : theBaseSelector(cfg, iC) {}

  const_iterator begin() const { return theSelectedTracks.begin(); }
  const_iterator end() const { return theSelectedTracks.end(); }
  size_t size() const { return theSelectedTracks.size(); }

  void select(const edm::Handle<reco::TrackCollection> &c, const edm::Event &evt, const edm::EventSetup &setup) {
    theSelectedTracks = theBaseSelector.select(c, evt, setup);
  }

private:
  container theSelectedTracks;

  AlignmentTrackFromVertexSelector theBaseSelector;
};

class AlignmentTrackFromVertexSelectorModule : public ObjectSelectorStream<TrackFromVertexConfigSelector> {
public:
  AlignmentTrackFromVertexSelectorModule(const edm::ParameterSet &ps)
      : ObjectSelectorStream<TrackFromVertexConfigSelector>(ps) {}
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Alignment Tracks Selector from Vertices");
    desc.add<edm::InputTag>("src", edm::InputTag("generalTracks"));
    desc.add<bool>("filter", false);
    desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
    desc.add<edm::InputTag>("leptonTracks", edm::InputTag("ALCARECOTkAlDiMuon"));
    desc.add<bool>("useClosestVertexToDilepton", false);
    desc.add<unsigned int>("vertexIndex", 0);
    descriptions.addWithDefaultLabel(desc);
  }
};

DEFINE_FWK_MODULE(AlignmentTrackFromVertexSelectorModule);
