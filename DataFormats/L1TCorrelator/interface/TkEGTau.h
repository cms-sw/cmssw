#ifndef DataFormatsL1TCorrelator_TkEGTau_h
#define DataFormatsL1TCorrelator_TkEGTau_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEGTau
//

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>

namespace l1t {

  class TkEGTau;

  typedef std::vector<TkEGTau> TkEGTauCollection;

  typedef edm::Ref<TkEGTauCollection> TkEGTauRef;
  typedef edm::RefVector<TkEGTauCollection> TkEGTauRefVector;
  typedef std::vector<TkEGTauRef> TkEGTauVectorRef;

  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollection;
  typedef edm::Ptr<L1TTTrackType> L1TTTrackRefPtr;
  typedef std::vector<L1TTTrackRefPtr> L1TTTrackRefPtr_Collection;
  typedef edm::Ref<EGammaBxCollection> EGammaRef;
  typedef std::vector<EGammaRef> EGammaVectorRef;

  class TkEGTau : public L1Candidate {
  public:
    TkEGTau();

    TkEGTau(const LorentzVector& p4,
            const std::vector<L1TTTrackRefPtr>& clustTracks,
            const std::vector<EGammaRef>& clustEGs,
            float iso = -999.);

    // ---------- const member functions ---------------------

    const L1TTTrackRefPtr seedTrk() const { return clustTracks_.at(0); }

    const std::vector<L1TTTrackRefPtr> trks() const { return clustTracks_; }

    const std::vector<EGammaRef> EGs() const { return clustEGs_; }

    float iso() const { return iso_; }

    // ---------- member functions ---------------------------

    void setVtxIso(float iso) { iso_ = iso; }

  private:
    std::vector<L1TTTrackRefPtr> clustTracks_;
    std::vector<EGammaRef> clustEGs_;
    float iso_;
  };
}  // namespace l1t

#endif
