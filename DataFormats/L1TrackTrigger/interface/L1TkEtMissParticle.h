#ifndef L1TkTrigger_L1TkEtMissParticle_h
#define L1TkTrigger_L1TkEtMissParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEtMissParticle
// 
/**\class L1TkEtMissParticle 

 Description: L1Extra particle class for EtMiss object.
*/
//
// Original Author:  E. Perez
//         Created:  Nov 14, 2013
//      
        
// system include files
        
// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"



namespace l1extra {

  class L1TkEtMissParticle : public reco::LeafCandidate
    { 
    public:
      
      enum EtMissType{ kMET, kMHT, kNumTypes } ;

      L1TkEtMissParticle();

      L1TkEtMissParticle(
        const LorentzVector& p4,
        EtMissType type,
        const double& etTotal,
	const double& etMissPU,
	const double& etTotalPU,
        const edm::Ref< L1TkPrimaryVertexCollection >& aVtxRef = edm::Ref< L1TkPrimaryVertexCollection >(),
        int bx = 0 ) ;



      // ---------- const member functions ---------------------

      EtMissType type() const { return type_ ; }  // kET or kHT

      // For type = kET, this is |MET|; for type = kHT, this is |MHT|
      double etMiss() const
        { return et() ; }

      // For type = kET, this is total ET; for type = kHT, this is total HT
      const double& etTotal() const
        { return etTot_ ; }

	// EtMiss and EtTot from PU vertices
	double etMissPU() const
	  { return etMissPU_ ; }
	double etTotalPU() const 
	  { return etTotalPU_ ; }

      int bx() const
        { return bx_ ; }

      const edm::Ref< L1TkPrimaryVertexCollection >& getVtxRef() const
        { return vtxRef_ ; }


      // ---------- member functions ---------------------------
      void setEtTotal( const double& etTotal )
        { etTot_ = etTotal ; }

      void setBx( int bx )
        { bx_ = bx ; }

    private:

      // ---------- member data --------------------------------

      EtMissType type_ ;
      double etTot_ ;
      double etMissPU_ ;
      double etTotalPU_ ;

      edm::Ref< L1TkPrimaryVertexCollection > vtxRef_ ;

      int bx_ ;

   };

}

#endif



