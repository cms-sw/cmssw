#ifndef DQM_TrackingMonitorSource_TrackAssociatorByDetId_h
#define DQM_TrackingMonitorSource_TrackAssociatorByDetId_h

// system includes
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// user includes
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Utilities/interface/EDMException.h"

// Associates two track collections ("monitored" and "reference") to each
// other, based on the number of valid RecHit DetIds they share.
//
// Both directions are computed in a single sweep:
//   - for each monitored track, the reference track sharing the most DetIds
//   - for each reference track, the monitored track sharing the most DetIds
// A pair is accepted if nShared / min(nMonitored, nReference) >= minSharedFraction.
// Using the smaller of the two hit counts makes the criterion symmetric, so
// that e.g. a short HLT track fully contained in a longer offline track is
// matched in both directions.
//
// Typical use inside a stream module:
//
//   TrackAssociatorByDetId assoc_{minSharedFraction};  // per-stream member
//
//   void Foo::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
//     auto const& mon = iEvent.get(monToken_);
//     auto const& ref = iEvent.get(refToken_);
//     assoc_.associate(mon, ref);
//     for (unsigned int i = 0; i < mon.size(); ++i) {
//       auto const& m = assoc_.monitoredToReference()[i];
//       if (m.matched) { /* ref[m.index], m.sharedFraction */ }
//     }
//   }
//
// Works with any random-access container of reco::Track (reco::TrackCollection,
// edm::View<reco::Track>). The tracks must have their TrackExtra and
// TrackingRecHits available (i.e. RECO/FEVT content, not AOD).
//
// Not thread-safe: the class keeps scratch buffers that are reused across
// events to avoid per-event allocations, so it is meant to be held as a
// per-stream member.
class TrackAssociatorByDetId {
public:
  // Best candidate for one track. nShared == 0 means that no track of the
  // other collection shares any DetId with it; `matched` tells whether the
  // candidate passes the shared-fraction requirement.
  struct Match {
    unsigned int index = 0;
    unsigned int nShared = 0;
    float sharedFraction = 0.f;
    bool matched = false;
  };
  using Matches = std::vector<Match>;

  explicit TrackAssociatorByDetId(double minSharedFraction) : minSharedFraction_(minSharedFraction) {}

  template <typename C>
  void associate(C const& monitored, C const& reference) {
    buildReferenceIndex(reference);

    const unsigned int nMon = monitored.size();
    const unsigned int nRef = reference.size();
    mon2ref_.assign(nMon, Match{});
    ref2mon_.assign(nRef, Match{});

    for (unsigned int iMon = 0; iMon < nMon; ++iMon) {
      extractDetIds(monitored[iMon], detIds_);
      const unsigned int nMonHits = detIds_.size();
      if (nMonHits == 0)
        continue;

      for (auto detId : detIds_) {
        auto range = std::equal_range(index_.begin(), index_.end(), Entry{detId, 0}, byDetId);
        for (auto it = range.first; it != range.second; ++it) {
          if (counts_[it->ref]++ == 0)
            touched_.push_back(it->ref);
        }
      }

      Match& monBest = mon2ref_[iMon];
      for (auto iRef : touched_) {
        const unsigned int nShared = counts_[iRef];
        counts_[iRef] = 0;
        const float frac = float(nShared) / std::min(nMonHits, nRefHits_[iRef]);
        if (nShared > monBest.nShared)
          monBest = Match{iRef, nShared, frac, frac >= minSharedFraction_};
        if (nShared > ref2mon_[iRef].nShared)
          ref2mon_[iRef] = Match{iMon, nShared, frac, frac >= minSharedFraction_};
      }
      touched_.clear();
    }
  }

  // results of the last associate() call, indexed as the input collections
  Matches const& monitoredToReference() const { return mon2ref_; }
  Matches const& referenceToMonitored() const { return ref2mon_; }

private:
  struct Entry {
    uint32_t detId;
    unsigned int ref;
  };
  static bool byDetId(Entry const& a, Entry const& b) { return a.detId < b.detId; }

  // Flat (detId, reference index) table sorted by detId: rebuilt every event
  // into a reused buffer, so no allocation happens once it reached its
  // high-water mark (unlike a map of vectors, whose nodes are freed by clear()).
  template <typename C>
  void buildReferenceIndex(C const& reference) {
    const unsigned int nRef = reference.size();
    index_.clear();
    nRefHits_.resize(nRef);
    counts_.assign(nRef, 0);
    touched_.clear();

    for (unsigned int iRef = 0; iRef < nRef; ++iRef) {
      extractDetIds(reference[iRef], detIds_);
      nRefHits_[iRef] = detIds_.size();
      for (auto detId : detIds_)
        index_.push_back(Entry{detId, iRef});
    }
    std::sort(index_.begin(), index_.end(), byDetId);
  }

  static void extractDetIds(reco::Track const& trk, std::vector<uint32_t>& buffer) {
    if (!trk.extra().isAvailable())
      throw edm::Exception(edm::errors::ProductNotFound)
          << "TrackAssociatorByDetId: the TrackExtra of the input tracks is not available;"
          << " hit-based association needs the TrackExtra and TrackingRecHit collections in the event.";

    buffer.clear();
    for (auto const& hit : trk.recHits()) {
      if (hit->isValid())
        buffer.push_back(hit->geographicalId().rawId());
    }
    // a DetId can appear twice (e.g. two clusters on the same module): count it once
    std::sort(buffer.begin(), buffer.end());
    buffer.erase(std::unique(buffer.begin(), buffer.end()), buffer.end());
  }

  const double minSharedFraction_;

  // ---- reference-side index, rebuilt each event
  std::vector<Entry> index_;
  std::vector<unsigned int> nRefHits_;

  // ---- results
  Matches mon2ref_;
  Matches ref2mon_;

  // ---- reused scratch buffers, per-stream
  std::vector<uint32_t> detIds_;
  std::vector<unsigned int> counts_;
  std::vector<unsigned int> touched_;
};

#endif
