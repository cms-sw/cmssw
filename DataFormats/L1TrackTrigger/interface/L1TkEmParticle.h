#ifndef L1TkTrigger_L1EmParticle_h
#define L1TkTrigger_L1EmParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


namespace l1t {
         
   class L1TkEmParticle : public reco::LeafCandidate
   {     
         
      public:
           
         L1TkEmParticle();

	 L1TkEmParticle( const LorentzVector& p4,
			    const edm::Ref< EGammaBxCollection >& egRef,
			    float tkisol = -999. );

	virtual ~L1TkEmParticle() {}

         // ---------- const member functions ---------------------

	 const edm::Ref< l1t::EGammaBxCollection >& getEGRef() const
	 { return egRef_ ; }

	 float getTrkIsol() const { return TrkIsol_ ; } 


         // ---------- member functions ---------------------------

	 void setTrkIsol(float TrkIsol)  { TrkIsol_ = TrkIsol ; }

     //	 int bx() const;

      private:

	 edm::Ref< l1t::EGammaBxCollection > egRef_ ;
	 float TrkIsol_;

    };
}

#endif


