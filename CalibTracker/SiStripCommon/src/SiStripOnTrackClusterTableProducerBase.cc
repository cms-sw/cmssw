#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"

#include "CalibTracker/SiStripCommon/interface/SiStripOnTrackClusterTableProducerBase.h"

SiStripOnTrackClusterTableProducerBase::~SiStripOnTrackClusterTableProducerBase() {}

namespace {
  int findTrackIndex(const edm::View<reco::Track>& tracks, const reco::Track* track) {
    for (auto iTr = tracks.begin(); iTr != tracks.end(); ++iTr) {
      if (&(*iTr) == track) {
        return iTr - tracks.begin();
      }
    }
    return -2;
  }
}  // namespace

void SiStripOnTrackClusterTableProducerBase::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::Track>> tracks;
  iEvent.getByToken(m_tracks_token, tracks);
  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociations;
  iEvent.getByToken(m_association_token, trajTrackAssociations);

  std::vector<OnTrackCluster> clusters{};

  for (const auto& assoc : *trajTrackAssociations) {
    const auto traj = assoc.key.get();
    const auto track = assoc.val.get();

    for (const auto& meas : traj->measurements()) {
      const auto& trajState = meas.updatedState();
      if (!trajState.isValid())
        continue;

      // there can be 2 (stereo module), 1 (no stereo module), or 0 (no strip hit) clusters per measurement
      const auto trechit = meas.recHit()->hit();
      const auto simple1d = dynamic_cast<const SiStripRecHit1D*>(trechit);
      const auto simple = dynamic_cast<const SiStripRecHit2D*>(trechit);
      const auto matched = dynamic_cast<const SiStripMatchedRecHit2D*>(trechit);
      if (matched) {
        clusters.emplace_back(matched->monoId(), &matched->monoCluster(), traj, track, meas);
        clusters.emplace_back(matched->stereoId(), &matched->stereoCluster(), traj, track, meas);
      } else if (simple) {
        clusters.emplace_back(simple->geographicalId().rawId(), simple->cluster().get(), traj, track, meas);
      } else if (simple1d) {
        clusters.emplace_back(simple1d->geographicalId().rawId(), simple1d->cluster().get(), traj, track, meas);
      }
    }
  }

  auto out = std::make_unique<nanoaod::FlatTable>(clusters.size(), m_name, false, m_extension);
  if (!m_extension) {
    std::vector<int> c_trackindex;
    c_trackindex.reserve(clusters.size());
    std::vector<uint32_t> c_rawid;
    c_rawid.reserve(clusters.size());
    for (const auto clus : clusters) {
      c_trackindex.push_back(findTrackIndex(*tracks, clus.track));
      c_rawid.push_back(clus.det);
    }
    addColumn(out.get(), "trackindex", c_trackindex, "Track index");
    addColumn(out.get(), "rawid", c_rawid, "DetId");
  }
  fillTable(clusters, *tracks, out.get(), iSetup);
  iEvent.put(std::move(out));
}
