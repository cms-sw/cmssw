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
// $Id: L1EtMissParticle.h,v 1.15 2009/03/22 16:11:30 wsun Exp $
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
      enum EtMissType{ kMET, kMHT, kNumTypes } ;

      L1EtMissParticle();

      // Default Refs are null.  For type = kET, only the first two are 
      // filled; for type = kHT, only the second two are filled.
      L1EtMissParticle(
	const LorentzVector& p4,
	EtMissType type,
	const double& etTotal,
	const edm::Ref< L1GctEtMissCollection >& aEtMissRef = edm::Ref< 
	L1GctEtMissCollection >(),
	const edm::Ref< L1GctEtTotalCollection >& aEtTotalRef = edm::Ref< 
	L1GctEtTotalCollection >(),
	const edm::Ref< L1GctHtMissCollection >& aHtMissRef = edm::Ref< 
	L1GctHtMissCollection >(),
	const edm::Ref< L1GctEtHadCollection >& aEtHadRef = edm::Ref< 
	L1GctEtHadCollection >(),
	int bx = 0 ) ;

      L1EtMissParticle(
	const PolarLorentzVector& p4,
	EtMissType type,
	const double& etTotal,
	const edm::Ref< L1GctEtMissCollection >& aEtMissRef = edm::Ref< 
	L1GctEtMissCollection >(),
	const edm::Ref< L1GctEtTotalCollection >& aEtTotalRef = edm::Ref< 
	L1GctEtTotalCollection >(),
	const edm::Ref< L1GctHtMissCollection >& aHtMissRef = edm::Ref< 
	L1GctHtMissCollection >(),
	const edm::Ref< L1GctEtHadCollection >& aEtHadRef = edm::Ref< 
	L1GctEtHadCollection >(),
	int bx = 0 ) ;

      virtual ~L1EtMissParticle() {}

      // ---------- const member functions ---------------------

      EtMissType type() const { return type_ ; }  // kET or kHT

      // For type = kET, this is |MET|; for type = kHT, this is |MHT|
      double etMiss() const
	{ return et() ; }

      // For type = kET, this is total ET; for type = kHT, this is total HT
      const double& etTotal() const
	{ return etTot_ ; }

      // This is filled only for type = kET
      const edm::Ref< L1GctEtMissCollection >& gctEtMissRef() const
	{ return etMissRef_ ; }

      // This is filled only for type = kET
      const edm::Ref< L1GctEtTotalCollection >& gctEtTotalRef() const
	{ return etTotRef_ ; }

      // This is filled only for type = kHT
      const edm::Ref< L1GctHtMissCollection >& gctHtMissRef() const
	{ return htMissRef_ ; }

      // This is filled only for type = kHT
      const edm::Ref< L1GctEtHadCollection >& gctEtHadRef() const
	{ return etHadRef_ ; }

      // This is filled only for type = kET
      const L1GctEtMiss* gctEtMiss() const
	{ return etMissRef_.get() ; }

      // This is filled only for type = kET
      const L1GctEtTotal* gctEtTotal() const
	{ return etTotRef_.get() ; }

      // This is filled only for type = kHT
      const L1GctHtMiss* gctHtMiss() const
	{ return htMissRef_.get() ; }

      // This is filled only for type = kHT
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

      void setBx( int bx )
	{ bx_ = bx ; }

    private:
      // L1EtMissParticle(const L1EtMissParticle&); // stop default

      // const L1EtMissParticle& operator=(const L1EtMissParticle&); // stop default

      // ---------- member data --------------------------------
      EtMissType type_ ;

      double etTot_ ;

      edm::Ref< L1GctEtMissCollection > etMissRef_ ;
      edm::Ref< L1GctEtTotalCollection > etTotRef_ ;
      edm::Ref< L1GctHtMissCollection > htMissRef_ ;
      edm::Ref< L1GctEtHadCollection > etHadRef_ ;

      int bx_ ;
    };
}

#endif
