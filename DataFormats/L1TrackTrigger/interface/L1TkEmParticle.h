#ifndef L1TkTrigger_L1EmParticle_h
#define L1TkTrigger_L1EmParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


namespace l1t {
         
   class L1TkEmParticle : public L1Candidate
   {     
         
      public:
           
         L1TkEmParticle();

	 L1TkEmParticle( const LorentzVector& p4,
			    const edm::Ref< EGammaBxCollection >& egRef,
			    float tkisol = -999. );

       L1TkEmParticle( const LorentzVector& p4,
                      const edm::Ref< EGammaBxCollection >& egRef,
                      float tkisol = -999. , float tkisolPV = -999);


	virtual ~L1TkEmParticle() {}

         // ---------- const member functions ---------------------

	 const edm::Ref< EGammaBxCollection >& getEGRef() const
	 { return egRef_ ; }

         const double l1RefEta() const
         { return egRef_->eta() ; }

         const double l1RefPhi() const
         { return egRef_->phi() ; }

         const double l1RefEt() const
         { return egRef_->et() ; }

	 float getTrkIsol() const { return TrkIsol_ ; } // not constrained to the PV, just track ptSum 

       float getTrkIsolPV() const { return TrkIsolPV_ ; } // constrained to the PV by DZ
 


         // ---------- member functions ---------------------------

	 void setTrkIsol(float TrkIsol)  { TrkIsol_ = TrkIsol ; }
       void setTrkIsolPV(float TrkIsolPV)  { TrkIsolPV_ = TrkIsolPV ; }

     //	 int bx() const;

      private:

	 edm::Ref< EGammaBxCollection > egRef_ ;
	 float TrkIsol_;
       float TrkIsolPV_;

    };
}

#endif


