#ifndef TkTrigger_TkEGTau_h
#define TkTrigger_TkEGTau_h

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

  typedef vector<TkEGTau> TkEGTauCollection;

  typedef edm::Ref<TkEGTauCollection> TkEGTauRef;
  typedef edm::RefVector<TkEGTauCollection> TkEGTauRefVector;
  typedef vector<TkEGTauRef> TkEGTauVectorRef;

  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef vector<L1TTTrackType> L1TTTrackCollection;
  typedef edm::Ptr<L1TTTrackType> L1TTTrackRefPtr;
  typedef vector<L1TTTrackRefPtr> L1TTTrackRefPtr_Collection;
  typedef edm::Ref<EGammaBxCollection> EGammaRef;
  typedef vector<EGammaRef> EGammaVectorRef;

  class TkEGTau : public L1Candidate {
  public:
    TkEGTau();

    TkEGTau(const LorentzVector& p4,
            const vector<L1TTTrackRefPtr>& clustTracks,
            const vector<EGammaRef>& clustEGs,
            float iso = -999.);

    virtual ~TkEGTau() {}

    // ---------- const member functions ---------------------

    const L1TTTrackRefPtr seedTrk() const { return clustTracks_.at(0); }

    const vector<L1TTTrackRefPtr> trks() const { return clustTracks_; }

    const vector<EGammaRef> EGs() const { return clustEGs_; }

    float iso() const { return iso_; }

    // ---------- member functions ---------------------------

    void setVtxIso(float iso) { iso_ = iso; }

  private:
    vector<L1TTTrackRefPtr> clustTracks_;
    vector<EGammaRef> clustEGs_;
    float iso_;
  };
}  // namespace l1t

#endif
