#ifndef DataFormatsL1TCorrelator_TkGlbMuon_h
#define DataFormatsL1TCorrelator_TkGlbMuon_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkGlbMuon

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1TCorrelator/interface/TkEm.h"

#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t {
  class TkGlbMuon : public L1Candidate {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
    typedef std::vector<L1TTTrackType> L1TTTrackCollection;

    TkGlbMuon() : theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}

    TkGlbMuon(const LorentzVector& p4,
              const edm::Ref<MuonBxCollection>& muRef,
              const edm::Ptr<L1TTTrackType>& trkPtr,
              float tkisol = -999.);

    //! more basic constructor, in case refs/ptrs can't be set or to be set separately
    TkGlbMuon(const L1Candidate& cand) : L1Candidate(cand), theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}

    const edm::Ptr<L1TTTrackType>& trkPtr() const { return trkPtr_; }

    const edm::Ref<MuonBxCollection>& muRef() const { return muRef_; }

    float trkIsol() const { return theIsolation; }
    float trkzVtx() const { return TrkzVtx_; }

    float dR() const { return dR_; }
    int nTracksMatched() const { return nTracksMatch_; }

    unsigned int quality() const { return quality_; }

    void setTrkPtr(const edm::Ptr<L1TTTrackType>& p) { trkPtr_ = p; }

    void setTrkzVtx(float TrkzVtx) { TrkzVtx_ = TrkzVtx; }
    void setTrkIsol(float TrkIsol) { theIsolation = TrkIsol; }

    void setdR(float dR) { dR_ = dR; }
    void setNTracksMatched(int nTracksMatch) { nTracksMatch_ = nTracksMatch; }

    void setQuality(unsigned int q) { quality_ = q; }

  private:
    // used for the Naive producer
    edm::Ref<MuonBxCollection> muRef_;

    edm::Ptr<L1TTTrackType> trkPtr_;

    float theIsolation;
    float TrkzVtx_;
    unsigned int quality_;
    float dR_;
    int nTracksMatch_;
  };
}  // namespace l1t

#endif
