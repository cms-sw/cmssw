#include "L1Trigger/VertexFinder/interface/VertexProducer.h"

using namespace l1tVertexFinder;
using namespace std;

VertexProducer::VertexProducer(const edm::ParameterSet& iConfig)
    : l1TracksToken_(consumes<TTTrackCollectionView>(iConfig.getParameter<edm::InputTag>("l1TracksInputTag"))),
      trackerTopologyToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      outputCollectionName_(iConfig.getParameter<std::string>("l1VertexCollectionName")),
      settings_(AlgoSettings(iConfig)) {
  // Get configuration parameters

  switch (settings_.vx_algo()) {
    case Algorithm::FastHisto:
      edm::LogInfo("VertexProducer") << "VertexProducer::Finding vertices using the FastHisto binning algorithm";
      break;
    case Algorithm::FastHistoLooseAssociation:
      edm::LogInfo("VertexProducer")
          << "VertexProducer::Finding vertices using the FastHistoLooseAssociation binning algorithm";
      break;
    case Algorithm::GapClustering:
      edm::LogInfo("VertexProducer") << "VertexProducer::Finding vertices using a gap clustering algorithm";
      break;
    case Algorithm::AgglomerativeHierarchical:
      edm::LogInfo("VertexProducer") << "VertexProducer::Finding vertices using a Simple Merge Clustering algorithm";
      break;
    case Algorithm::DBSCAN:
      edm::LogInfo("VertexProducer") << "VertexProducer::Finding vertices using a DBSCAN algorithm";
      break;
    case Algorithm::PVR:
      edm::LogInfo("VertexProducer") << "VertexProducer::Finding vertices using a PVR algorithm";
      break;
    case Algorithm::AdaptiveVertexReconstruction:
      edm::LogInfo("VertexProducer")
          << "VertexProducer::Finding vertices using an AdaptiveVertexReconstruction algorithm";
      break;
    case Algorithm::HPV:
      edm::LogInfo("VertexProducer") << "VertexProducer::Finding vertices using a Highest Pt Vertex algorithm";
      break;
    case Algorithm::Kmeans:
      edm::LogInfo("VertexProducer") << "VertexProducer::Finding vertices using a kmeans algorithm";
      break;
  }

  // Tame debug printout.
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(4);

  //--- Define EDM output to be written to file (if required)
  produces<l1t::VertexCollection>(outputCollectionName_);
}

void VertexProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}

void VertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TTTrackCollectionView> l1TracksHandle;
  iEvent.getByToken(l1TracksToken_, l1TracksHandle);

  std::vector<L1Track> l1Tracks;
  l1Tracks.reserve(l1TracksHandle->size());

  for (const auto& track : l1TracksHandle->ptrs())
    l1Tracks.push_back(L1Track(track));

  std::vector<const L1Track*> l1TrackPtrs;
  l1TrackPtrs.reserve(l1Tracks.size());
  for (const auto& track : l1Tracks) {
    if (track.pt() > settings_.vx_TrackMinPt()) {
      if (track.pt() < 50 or track.getNumStubs() > 5)
        l1TrackPtrs.push_back(&track);
    }
  }

  // FIXME: Check with Davide if the tracks should be filtered using the following cuts
  //   fittedTracks[i].second.accepted() and fittedTracks[i].second.chi2dof()< settings_->chi2OverNdfCut()
  VertexFinder vf(l1TrackPtrs, settings_);

  switch (settings_.vx_algo()) {
    case Algorithm::FastHisto: {
      edm::ESHandle<TrackerTopology> tTopoHandle = iSetup.getHandle(trackerTopologyToken_);
      vf.FastHisto(tTopoHandle.product());
      break;
    }
    case Algorithm::FastHistoLooseAssociation:
      vf.FastHistoLooseAssociation();
      break;
    case Algorithm::GapClustering:
      vf.GapClustering();
      break;
    case Algorithm::AgglomerativeHierarchical:
      vf.AgglomerativeHierarchicalClustering();
      break;
    case Algorithm::DBSCAN:
      vf.DBSCAN();
      break;
    case Algorithm::PVR:
      vf.PVR();
      break;
    case Algorithm::AdaptiveVertexReconstruction:
      vf.AdaptiveVertexReconstruction();
      break;
    case Algorithm::HPV:
      vf.HPV();
      break;
    case Algorithm::Kmeans:
      vf.Kmeans();
      break;
  }

  vf.SortVerticesInZ0();
  vf.FindPrimaryVertex();

  // //=== Store output EDM track and hardware stub collections.
  std::unique_ptr<l1t::VertexCollection> lProduct(new std::vector<l1t::Vertex>());

  for (const auto& vtx : vf.Vertices()) {
    std::vector<edm::Ptr<l1t::Vertex::Track_t>> lVtxTracks;
    lVtxTracks.reserve(vtx.tracks().size());
    for (const auto& t : vtx.tracks())
      lVtxTracks.push_back(t->getTTTrackPtr());
    lProduct->emplace_back(l1t::Vertex(vtx.z0(), lVtxTracks));
  }
  iEvent.put(std::move(lProduct), outputCollectionName_);
}

void VertexProducer::endJob() {}

DEFINE_FWK_MODULE(VertexProducer);
