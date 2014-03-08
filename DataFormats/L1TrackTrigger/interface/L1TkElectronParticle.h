#ifndef L1TkTrigger_L1ElectronParticle_h
#define L1TkTrigger_L1ElectronParticle_h

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

//#include "SimDataFormats/SLHC/interface/L1TkTrack.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


namespace l1extra {
         
   class L1TkElectronParticle : public L1TkEmParticle
   {     
         
      public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >   L1TkTrackCollectionType;

           
         L1TkElectronParticle();

	 L1TkElectronParticle( const LorentzVector& p4,
			    const edm::Ref< L1EmParticleCollection >& egRef,
			    const edm::Ptr< L1TkTrackType >& trkPtr,
			    float tkisol = -999. );

	virtual ~L1TkElectronParticle() {}

         // ---------- const member functions ---------------------

         const edm::Ptr< L1TkTrackType >& getTrkPtr() const
         { return trkPtr_ ; }

 	 float getTrkzVtx() const { return TrkzVtx_ ; }


         // ---------- member functions ---------------------------

	 void setTrkzVtx(float TrkzVtx)  { TrkzVtx_ = TrkzVtx ; }

      private:

         edm::Ptr< L1TkTrackType > trkPtr_ ;
	 float TrkzVtx_ ;

    };
}

#endif


