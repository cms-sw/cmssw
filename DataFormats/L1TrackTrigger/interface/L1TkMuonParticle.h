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
#include "DataFormats/L1Trigger/interface/L1MuonParticleExtended.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleExtendedFwd.h"

namespace l1extra
{
  class L1TkMuonParticle : public reco::LeafCandidate
  {
    public:

      typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
      typedef std::vector< L1TkTrackType >   L1TkTrackCollectionType;

      L1TkMuonParticle() : theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}
      L1TkMuonParticle( const LorentzVector& p4,
                        const edm::Ptr< DTMatch >& muRef,
                        float tkisol = -999. );

      L1TkMuonParticle( const LorentzVector& p4,
   		        const edm::Ref< L1MuonParticleCollection >& muRef,
		        const edm::Ptr< L1TkTrackType >& trkPtr,
		        float tkisol = -999. );

      //! more basic constructor, in case refs/ptrs can't be set or to be set separately
      L1TkMuonParticle(const reco::LeafCandidate& cand) : reco::LeafCandidate(cand), theIsolation(-999.), TrkzVtx_(999.), quality_(999) {}

      virtual ~L1TkMuonParticle() {}

      // const member functions
      const edm::Ptr< DTMatch >& getDTMatchPtr() const { return theDTMatch; }

      const edm::Ptr< L1TkTrackType >& getTrkPtr() const
         { return trkPtr_ ; }

      const L1MuonParticleRef&  getMuRef() const
	{ return muRef_ ; }

      const L1MuonParticleExtendedRef&  getMuExtendedRef() const
	{ return muExtendedRef_ ; }
    
      float getTrkIsol() const { return theIsolation; }
      float getTrkzVtx() const { return TrkzVtx_ ; }

      int bx() const ;

      unsigned int quality()  const {return quality_;}

      void setTrkPtr(const edm::Ptr< L1TkTrackType >& p) {trkPtr_ = p;}
      void setMuRef(const L1MuonParticleRef& r){muRef_ = r;}
      void setMuExtendedRef(const L1MuonParticleExtendedRef& r){muExtendedRef_ = r;}
      
      void setTrkzVtx(float TrkzVtx) { TrkzVtx_ = TrkzVtx ; }
      void setTrkIsol(float TrkIsol) { theIsolation = TrkIsol ; }

      void setQuality(unsigned int q){ quality_ = q;}

    private:


	// used for Padova's muons:
      edm::Ptr< DTMatch > theDTMatch;

	// used for the Naive producer
      edm::Ref< L1MuonParticleCollection > muRef_ ;

	// used for Slava's muons:
      L1MuonParticleExtendedRef muExtendedRef_ ;

      edm::Ptr< L1TkTrackType > trkPtr_ ;

      float theIsolation;
      float TrkzVtx_ ;
      unsigned int quality_;
  };
}

#endif

