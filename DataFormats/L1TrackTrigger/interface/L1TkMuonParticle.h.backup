#ifndef L1TkTrigger_L1MuonParticle_h
#define L1TkTrigger_L1MuonParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

// PL :include PierLuigi's class
// #include "....."

namespace l1extra {
         
   class L1TkMuonParticle : public reco::LeafCandidate
   {     
         
      public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >   L1TkTrackCollectionType;
           
         L1TkMuonParticle();

	 L1TkMuonParticle( const LorentzVector& p4,
			    // const edm::Ref< XXXCollection >& muRef,     // reference to PL's object
			     const edm::Ref< L1MuonParticleCollection >& muRef,
			    const edm::Ptr< L1TkTrackType >& trkPtr,
			    float tkisol = -999. );

	virtual ~L1TkMuonParticle() {}

         // ---------- const member functions ---------------------

         const edm::Ptr< L1TkTrackType >& getTrkPtr() const
         { return trkPtr_ ; }

	// PL :
         // const edm::Ref< XXXCollection >& getMuRef() const
         // { return muRef_ ; }

 	 float getTrkzVtx() const { return TrkzVtx_ ; }
         float getTrkIsol() const { return TrkIsol_ ; }


         // ---------- member functions ---------------------------

	 void setTrkzVtx(float TrkzVtx)  { TrkzVtx_ = TrkzVtx ; }
         void setTrkIsol(float TrkIsol)  { TrkIsol_ = TrkIsol ; }
         int bx() const;

	  //void setDeltaR(float dr) { DeltaR_ = dr ; }

      private:

	// PL
         // edm::Ref< XXXCollection > muRef_ ;
	 edm::Ref< L1MuonParticleCollection > muRef_ ;

         edm::Ptr< L1TkTrackType > trkPtr_ ;

         float TrkIsol_;
	 float TrkzVtx_ ;
	
	 //float DeltaR_ ;	// temporary

    };
}

#endif


