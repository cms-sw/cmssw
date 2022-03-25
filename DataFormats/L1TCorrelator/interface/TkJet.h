#ifndef DataFormatsL1TCorrelator_TkJet_h
#define DataFormatsL1TCorrelator_TkJet_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkJet
//

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace l1t {

  class TkJet : public L1Candidate {
  public:
    typedef TTTrack<Ref_Phase2TrackerDigi_> L1TTTrackType;
    typedef std::vector<L1TTTrackType> L1TTTrackCollection;

    TkJet();

    TkJet(const LorentzVector& p4,
          const edm::Ref<JetBxCollection>& jetRef,
          const std::vector<edm::Ptr<L1TTTrackType> >& trkPtrs,
          float jetvtx = -999.);
    TkJet(const LorentzVector& p4,
          const std::vector<edm::Ptr<L1TTTrackType> >& trkPtrs,
          float jetvtx = -999.,
          unsigned int ntracks = 0,
          unsigned int tighttracks = 0,
          unsigned int displacedtracks = 0,
          unsigned int tightdisplacedtracks = 0,
          bool displacedTag = false);

    // ---------- const member functions ---------------------

    const edm::Ref<JetBxCollection>& jetRef() const { return jetRef_; }

    const std::vector<edm::Ptr<L1TTTrackType> >& trkPtrs() const { return trkPtrs_; }

    float jetVtx() const { return JetVtx_; }
    unsigned int ntracks() const { return ntracks_; }
    unsigned int nTighttracks() const { return tighttracks_; }
    unsigned int nDisptracks() const { return displacedtracks_; }
    unsigned int nTightDisptracks() const { return tightdisplacedtracks_; }
    bool isDisplaced() const { return displacedTag_; }

    // ---------- member functions ---------------------------
    void setJetVtx(float JetVtx) { JetVtx_ = JetVtx; }

    int bx() const;

  private:
    edm::Ref<JetBxCollection> jetRef_;
    std::vector<edm::Ptr<L1TTTrackType> > trkPtrs_;
    float JetVtx_;
    unsigned int ntracks_, tighttracks_, displacedtracks_, tightdisplacedtracks_;
    bool displacedTag_;
  };
}  // namespace l1t

#endif
