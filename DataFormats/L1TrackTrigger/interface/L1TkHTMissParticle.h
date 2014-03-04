#ifndef L1TkTrigger_L1TkHTMissParticle_h
#define L1TkTrigger_L1TkHTMissParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkHTMissParticle
// 
/**\class L1TkHTMissParticle 

 Description: L1Extra particle class for HTMiss object.
*/
//
// Original Author:  E. Perez
//         Created:  Nov 14, 2013
//      
        
// system include files
        
// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleFwd.h"




namespace l1extra {

  class L1TkHTMissParticle : public reco::LeafCandidate
    { 
    public:
      
      L1TkHTMissParticle();

      L1TkHTMissParticle(
        const LorentzVector& p4,
        const double& EtTotal,
        const edm::RefProd< L1TkJetParticleCollection >& jetCollRef = edm::RefProd< L1TkJetParticleCollection >(),
        const edm::Ref< L1TkPrimaryVertexCollection >& aVtxRef = edm::Ref< L1TkPrimaryVertexCollection >(),
        int bx = 0 ) ;



      // ---------- const member functions ---------------------


      double EtMiss() const
        { return et() ; }	// HTM (missing HT)

      const double& EtTotal() const
        { return EtTot_ ; }	

	// HTM and HT from PU vertices
	double EtMissPU() const
	  { return EtMissPU_ ; }

	double EtTotalPU() const 
	  { return EtTotalPU_ ; }

      int bx() const
        { return bx_ ; }

      float getVtx()  const 
	{ return zvtx_ ; }

      const edm::RefProd< L1TkJetParticleCollection >& getjetCollectionRef() const
        { return jetCollectionRef_ ; }

      const edm::Ref< L1TkPrimaryVertexCollection >& getVtxRef() const
	{ return vtxRef_ ;}

      // ---------- member functions ---------------------------
      void setEtTotal( const double& EtTotal )
        { EtTot_ = EtTotal ; }

      void setEtTotalPU( const double& EtTotalPU )
	{ EtTotalPU_ = EtTotalPU ; }

      void setEtMissPU( const double& EtMissPU ) 
	{ EtMissPU_ = EtMissPU ; }

      void setVtx( const float& zvtx )
	{ zvtx_ = zvtx ; }

      void setBx( int bx )
        { bx_ = bx ; }

    private:

      // ---------- member data --------------------------------

      float zvtx_;		// zvtx used to constrain the jets
      double EtTot_ ;		// HT
      double EtMissPU_ ;	// HTM form jets that don't come from zvtx
      double EtTotalPU_ ;	// HT from jets that don't come from zvtx

	// reference to the collection of L1TkJets used to calculate HT and HTM :
      edm::RefProd< L1TkJetParticleCollection > jetCollectionRef_ ;

	// optional. In case the TkJets are constrained to the event primary vertex,
	// instead of using the jet vertices.
      edm::Ref< L1TkPrimaryVertexCollection > vtxRef_ ;


      int bx_ ;

   };

}

#endif



