#ifndef L1TkTrigger_L1TauParticle_h
#define L1TkTrigger_L1TauParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkTauParticle
//

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"


namespace l1extra {
         
   class L1TkTauParticle : public reco::LeafCandidate
   {     
         
      public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >   L1TkTrackCollectionType;
           
         L1TkTauParticle();

	 L1TkTauParticle( const LorentzVector& p4,
			    const edm::Ref< L1JetParticleCollection >& tauCaloRef,   // null for stand-alone TkTaus
			    const edm::Ptr< L1TkTrackType >& trkPtr,
                            const edm::Ptr< L1TkTrackType >& trkPtr2,	// null for tau -> 1 prong
                            const edm::Ptr< L1TkTrackType >& trkPtr3,	// null for tau -> 1 prong
			    float tkisol = -999. );

	virtual ~L1TkTauParticle() {}

         // ---------- const member functions ---------------------

         const edm::Ref< L1JetParticleCollection >& gettauCaloRef() const
	  { return tauCaloRef_  ; }

         const edm::Ptr< L1TkTrackType >& getTrkPtr() const
         { return trkPtr_ ; }

         const edm::Ptr< L1TkTrackType >& getTrkPtr2() const
         { return trkPtr2_ ; }
         const edm::Ptr< L1TkTrackType >& getTrkPtr3() const
         { return trkPtr3_ ; }

 	 float getTrkzVtx() const { return TrkzVtx_ ; }
         float getTrkIsol() const { return TrkIsol_ ; }


         // ---------- member functions ---------------------------

	 void setTrkzVtx(float TrkzVtx)  { TrkzVtx_ = TrkzVtx ; }
         void setTrkIsol(float TrkIsol)  { TrkIsol_ = TrkIsol ; }
         int bx() const;

      private:

	 edm::Ref< L1JetParticleCollection > tauCaloRef_ ;

         edm::Ptr< L1TkTrackType > trkPtr_ ;
         edm::Ptr< L1TkTrackType > trkPtr2_ ;
         edm::Ptr< L1TkTrackType > trkPtr3_ ;

         float TrkIsol_;
	 float TrkzVtx_ ;
	
    };
}

#endif


