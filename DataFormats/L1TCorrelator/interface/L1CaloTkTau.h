#ifndef DataFormatsL1TCorrelator_TkCaloTkTau_h
#define DataFormatsL1TCorrelator_TkCaloTkTau_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1CaloTkTau
//

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>

namespace l1t {

  class L1CaloTkTau;

  typedef std::vector<L1CaloTkTau> L1CaloTkTauCollection;

  typedef edm::Ref<L1CaloTkTauCollection> L1CaloTkTauRef;
  typedef edm::RefVector<L1CaloTkTauCollection> L1CaloTkTauRefVector;
  typedef std::vector<L1CaloTkTauRef> L1CaloTkTauVectorRef;

  typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
  typedef std::vector<L1TTTrackType> L1TTTrackCollection;
  typedef edm::Ptr<L1TTTrackType> L1TTTrackRefPtr;
  typedef std::vector<L1TTTrackRefPtr> L1TTTrackRefPtr_Collection;

  class L1CaloTkTau : public L1Candidate {
  public:
    L1CaloTkTau();

    L1CaloTkTau(const LorentzVector& p4,  // caloTau calibrated p4
                const LorentzVector& tracksP4,
                const std::vector<L1TTTrackRefPtr>& clustTracks,
                Tau& caloTau,
                float vtxIso = -999.);

    // ---------- const member functions ---------------------

    const L1TTTrackRefPtr seedTrk() const { return clustTracks_.at(0); }

    float vtxIso() const { return vtxIso_; }

    LorentzVector trackBasedP4() const { return tracksP4_; }

    float trackBasedEt() const { return tracksP4_.Et(); }

    Tau caloTau() const { return caloTau_; }

    // ---------- member functions ---------------------------

    void setVtxIso(float VtxIso) { vtxIso_ = VtxIso; }

  private:
    LorentzVector tracksP4_;
    std::vector<L1TTTrackRefPtr> clustTracks_;
    Tau caloTau_;
    float vtxIso_;
  };
}  // namespace l1t

#endif
