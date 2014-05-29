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

      L1TkMuonParticle( const LorentzVector& p4,
   		        const edm::Ref< L1MuonParticleCollection >& muRef,
		        const edm::Ptr< L1TkTrackType >& trkPtr,
		        float tkisol = -999. );

      virtual ~L1TkMuonParticle() {}

      // const member functions
      const edm::Ptr< DTMatch >& getDTMatchPtr() const { return theDTMatch; }

      const edm::Ptr< L1TkTrackType >& getTrkPtr() const
         { return trkPtr_ ; }

      float getTrkIsol() const { return theIsolation; }
      float getTrkzVtx() const { return TrkzVtx_ ; }

      int bx() const ;

	void setTrkzVtx(float TrkzVtx) { TrkzVtx_ = TrkzVtx ; }
        void setTrkIsol(float TrkIsol) { theIsolation = TrkIsol ; }

        unsigned int quality()  const;


    private:


	// used for Padova's muons:
      edm::Ptr< DTMatch > theDTMatch;

	// used for the Naive producer
      edm::Ref< L1MuonParticleCollection > muRef_ ;
      edm::Ptr< L1TkTrackType > trkPtr_ ;

      float theIsolation;
      float TrkzVtx_ ;

  };
}

#endif

