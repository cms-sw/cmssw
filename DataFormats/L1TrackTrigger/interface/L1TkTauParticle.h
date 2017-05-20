#ifndef L1TkTrigger_L1TauParticle_h
#define L1TkTrigger_L1TauParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkTauParticle
//

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"

#include "DataFormats/L1Trigger/interface/Tau.h"


namespace l1t {
         
   class L1TkTauParticle : public L1Candidate
   {     
         
      public:

    typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
    typedef std::vector< L1TTTrackType > L1TTTrackCollection;
           
         L1TkTauParticle();

	 L1TkTauParticle( const LorentzVector& p4,
			    const edm::Ref< TauBxCollection >& tauCaloRef,   // null for stand-alone TkTaus
			    const edm::Ptr< L1TTTrackType >& trkPtr,
                            const edm::Ptr< L1TTTrackType >& trkPtr2,	// null for tau -> 1 prong
                            const edm::Ptr< L1TTTrackType >& trkPtr3,	// null for tau -> 1 prong
			    float tkisol = -999. );

	virtual ~L1TkTauParticle() {}

         // ---------- const member functions ---------------------

         const edm::Ref< TauBxCollection >& gettauCaloRef() const
	  { return tauCaloRef_  ; }

         const edm::Ptr< L1TTTrackType >& getTrkPtr() const
         { return trkPtr_ ; }

         const edm::Ptr< L1TTTrackType >& getTrkPtr2() const
         { return trkPtr2_ ; }
         const edm::Ptr< L1TTTrackType >& getTrkPtr3() const
         { return trkPtr3_ ; }

 	 float getTrkzVtx() const { return TrkzVtx_ ; }
         float getTrkIsol() const { return TrkIsol_ ; }


         // ---------- member functions ---------------------------

	 void setTrkzVtx(float TrkzVtx)  { TrkzVtx_ = TrkzVtx ; }
         void setTrkIsol(float TrkIsol)  { TrkIsol_ = TrkIsol ; }
         int bx() const;

      private:

	 edm::Ref< TauBxCollection > tauCaloRef_ ;

         edm::Ptr< L1TTTrackType > trkPtr_ ;
         edm::Ptr< L1TTTrackType > trkPtr2_ ;
         edm::Ptr< L1TTTrackType > trkPtr3_ ;

         float TrkIsol_;
	 float TrkzVtx_ ;
	
    };
}

#endif


