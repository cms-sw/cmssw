#ifndef L1TkTrigger_L1JetParticle_h
#define L1TkTrigger_L1JetParticle_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkJetParticle
// 

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"


namespace l1extra {

   class L1TkJetParticle : public reco::LeafCandidate
   {

      public:

         L1TkJetParticle();

         L1TkJetParticle( const LorentzVector& p4,
                            const edm::Ref< L1JetParticleCollection >& jetRef,
			    float jetvtx = -999. );

        virtual ~L1TkJetParticle() {}

         // ---------- const member functions ---------------------

         const edm::Ref< L1JetParticleCollection >& getJetRef() const
         { return jetRef_ ; }

	 float getJetVtx() const  { return JetVtx_ ; }


         // ---------- member functions ---------------------------

	 void setJetVtx(float JetVtx) { JetVtx_ = JetVtx ; }

         int bx() const;

      private:
         edm::Ref< L1JetParticleCollection > jetRef_ ;
	 float JetVtx_ ;

    };
}

#endif

