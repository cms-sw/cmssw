#ifndef L1TkTrigger_L1MuonParticle_h
#define L1TkTrigger_L1MuonParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkMuonParticle

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTBtiTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTTSPhiTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTTSThetaTrigger.h"
#include "DataFormats/L1DTPlusTrackTrigger/interface/DTMatch.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

namespace l1extra
{
  class L1TkMuonParticle : public reco::LeafCandidate
  {
    public:

      typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
      typedef std::vector< L1TkTrackType >   L1TkTrackCollectionType;

      L1TkMuonParticle();
      L1TkMuonParticle( const LorentzVector& p4,
                        const edm::Ptr< DTMatch >& muRef,
                        float tkisol = -999. );
      virtual ~L1TkMuonParticle() {}

      // const member functions
      const edm::Ptr< DTMatch >& getDTMatchPtr() const { return theDTMatch; }
      float getTrkIsol() const { return theIsolation; }

      int bx() const { return theDTMatch->getDTBX(); }

    private:

      edm::Ptr< DTMatch > theDTMatch;
      float theIsolation;

  };
}

#endif

