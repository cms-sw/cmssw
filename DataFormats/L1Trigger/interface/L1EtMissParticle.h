#ifndef L1Trigger_L1EtMissParticle_h
#define L1Trigger_L1EtMissParticle_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1EtMissParticle
// 
/**\class L1EtMissParticle \file L1EtMissParticle.h DataFormats/L1Trigger/interface/L1EtMissParticle.h \author Werner Sun

 Description: L1Extra particle class for EtMiss object.
*/
//
// Original Author:  Werner Sun
//         Created:  Sat Jul 15 12:41:07 EDT 2006
// $Id: L1EtMissParticle.h,v 1.12 2007/11/13 17:27:23 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/Common/interface/Ref.h"

// forward declarations

namespace l1extra {

   class L1EtMissParticle : public reco::LeafCandidate
   {

      public:
	 L1EtMissParticle();

	 // Default Refs are null.
	 L1EtMissParticle(
            const LorentzVector& p4,
	    const double& etTotal,
	    const double& etHad,
	    const edm::Ref< L1GctEtMissCollection >& aEtMissRef =
	       edm::Ref< L1GctEtMissCollection >(),
	    const edm::Ref< L1GctEtTotalCollection >& aEtTotalRef =
	       edm::Ref< L1GctEtTotalCollection >(),
	    const edm::Ref< L1GctEtHadCollection >& aEtHadRef =
	       edm::Ref< L1GctEtHadCollection >(),
	    int bx = 0 ) ;

	 L1EtMissParticle(
	    const PolarLorentzVector& p4,
	    const double& etTotal,
	    const double& etHad,
	    const edm::Ref< L1GctEtMissCollection >& aEtMissRef =
	       edm::Ref< L1GctEtMissCollection >(),
	    const edm::Ref< L1GctEtTotalCollection >& aEtTotalRef =
	       edm::Ref< L1GctEtTotalCollection >(),
	    const edm::Ref< L1GctEtHadCollection >& aEtHadRef =
	       edm::Ref< L1GctEtHadCollection >(),
	    int bx = 0 ) ;

	 virtual ~L1EtMissParticle() {}

	 // ---------- const member functions ---------------------
	 double etMiss() const
	 { return et() ; }

	 const double& etTotal() const
	 { return etTot_ ; }

	 const double& etHad() const
	 { return etHad_ ; }

	 const edm::Ref< L1GctEtMissCollection >& gctEtMissRef() const
	 { return etMissRef_ ; }

	 const edm::Ref< L1GctEtTotalCollection >& gctEtTotalRef() const
	 { return etTotRef_ ; }

	 const edm::Ref< L1GctEtHadCollection >& gctEtHadRef() const
	 { return etHadRef_ ; }

	 const L1GctEtMiss* gctEtMiss() const
	 { return etMissRef_.get() ; }

	 const L1GctEtTotal* gctEtTotal() const
	 { return etTotRef_.get() ; }

	 const L1GctEtHad* gctEtHad() const
	 { return etHadRef_.get() ; }

         virtual L1EtMissParticle* clone() const
         { return new L1EtMissParticle( *this ) ; }

	 int bx() const
	 { return bx_ ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------
	 void setEtTotal( const double& etTotal )
	 { etTot_ = etTotal ; }

	 void setEtHad( const double& etHad )
	 { etHad_ = etHad ; }

	 void setBx( int bx )
	 { bx_ = bx ; }

      private:
	 // L1EtMissParticle(const L1EtMissParticle&); // stop default

	 // const L1EtMissParticle& operator=(const L1EtMissParticle&); // stop default

	 // ---------- member data --------------------------------
	 double etTot_ ;
	 double etHad_ ;

	 edm::Ref< L1GctEtMissCollection > etMissRef_ ;
	 edm::Ref< L1GctEtTotalCollection > etTotRef_ ;
	 edm::Ref< L1GctEtHadCollection > etHadRef_ ;

	 int bx_ ;
   };
}

#endif
