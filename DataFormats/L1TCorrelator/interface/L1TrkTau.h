#ifndef DataFormatsL1TCorrelator_TkTrkTau_h
#define DataFormatsL1TCorrelator_TkTrkTau_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TrkTau
//

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>

namespace l1t {

  class L1TrkTau;

  typedef std::vector<L1TrkTau> L1TrkTauCollection;

  typedef edm::Ref<L1TrkTauCollection> L1TrkTauRef;
  typedef edm::RefVector<L1TrkTauCollection> L1TrkTauRefVector;
  typedef std::vector<L1TrkTauRef> L1TrkTauVectorRef;

  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollection;
  typedef edm::Ptr<L1TTTrackType> L1TTTrackRefPtr;
  typedef std::vector<L1TTTrackRefPtr> L1TTTrackRefPtr_Collection;

  class L1TrkTau : public L1Candidate {
  public:
    L1TrkTau();

    L1TrkTau(const LorentzVector& p4, const std::vector<L1TTTrackRefPtr>& clustTracks, float iso = -999.);

    // ---------- const member functions ---------------------

    const L1TTTrackRefPtr seedTrk() const { return clustTracks_.at(0); }

    const std::vector<L1TTTrackRefPtr> trks() const { return clustTracks_; }

    float iso() const { return iso_; }

    // ---------- member functions ---------------------------

    void setIso(float iso) { iso_ = iso; }

  private:
    std::vector<L1TTTrackRefPtr> clustTracks_;
    float iso_;
  };
}  // namespace l1t

#endif
