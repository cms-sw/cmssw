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
// $Id: L1EtMissParticle.h,v 1.11 2007/11/13 03:07:45 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"
#include "DataFormats/Common/interface/RefProd.h"

// forward declarations

namespace l1extra {

   class L1EtMissParticle : public reco::LeafCandidate
   {

      public:
	 L1EtMissParticle();

	 // Default Refs are null.
	 L1EtMissParticle( const LorentzVector& p4,
			   const double& etTotal,
			   const double& etHad,
			   const edm::RefProd< L1GctEtMiss >& aEtMissRef =
			      edm::RefProd< L1GctEtMiss >(),
			   const edm::RefProd< L1GctEtTotal >& aEtTotalRef =
			      edm::RefProd< L1GctEtTotal >(),
			   const edm::RefProd< L1GctEtHad >& aEtHadRef =
			      edm::RefProd< L1GctEtHad >() ) ;

	 L1EtMissParticle( const PolarLorentzVector& p4,
			   const double& etTotal,
			   const double& etHad,
			   const edm::RefProd< L1GctEtMiss >& aEtMissRef =
			      edm::RefProd< L1GctEtMiss >(),
			   const edm::RefProd< L1GctEtTotal >& aEtTotalRef =
			      edm::RefProd< L1GctEtTotal >(),
			   const edm::RefProd< L1GctEtHad >& aEtHadRef =
			      edm::RefProd< L1GctEtHad >() ) ;

	 virtual ~L1EtMissParticle() {}

	 // ---------- const member functions ---------------------
	 double etMiss() const
	 { return et() ; }

	 const double& etTotal() const
	 { return etTot_ ; }

	 const double& etHad() const
	 { return etHad_ ; }

	 const edm::RefProd< L1GctEtMiss >& gctEtMissRef() const
	 { return etMissRef_ ; }

	 const edm::RefProd< L1GctEtTotal >& gctEtTotalRef() const
	 { return etTotRef_ ; }

	 const edm::RefProd< L1GctEtHad >& gctEtHadRef() const
	 { return etHadRef_ ; }

	 const L1GctEtMiss* gctEtMiss() const
	 { return etMissRef_.get() ; }

	 const L1GctEtTotal* gctEtTotal() const
	 { return etTotRef_.get() ; }

	 const L1GctEtHad* gctEtHad() const
	 { return etHadRef_.get() ; }

         virtual L1EtMissParticle* clone() const
         { return new L1EtMissParticle( *this ) ; }

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------
	 void setEtTotal( const double& etTotal )
	 { etTot_ = etTotal ; }

	 void setEtHad( const double& etHad )
	 { etHad_ = etHad ; }

      private:
	 // L1EtMissParticle(const L1EtMissParticle&); // stop default

	 // const L1EtMissParticle& operator=(const L1EtMissParticle&); // stop default

	 // ---------- member data --------------------------------
	 double etTot_ ;
	 double etHad_ ;

	 edm::RefProd< L1GctEtMiss > etMissRef_ ;
	 edm::RefProd< L1GctEtTotal > etTotRef_ ;
	 edm::RefProd< L1GctEtHad > etHadRef_ ;
   };
}

#endif
