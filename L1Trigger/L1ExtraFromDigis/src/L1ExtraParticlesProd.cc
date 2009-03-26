// -*- C++ -*-
//
// Package:    L1ExtraParticlesProd
// Class:      L1ExtraParticlesProd
// 
/**\class L1ExtraParticlesProd \file L1ExtraParticlesProd.cc src/L1ExtraParticlesProd/src/L1ExtraParticlesProd.cc
*/
//
// Original Author:  Werner Sun
//         Created:  Mon Oct  2 22:45:32 EDT 2006
// $Id: L1ExtraParticlesProd.cc,v 1.23 2009/03/26 03:58:28 wsun Exp $
//
//


// system include files
#include <memory>

// user include files
#include "L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticlesProd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #include "FWCore/Utilities/interface/EDMException.h"

//
// class decleration
//


//
// constants, enums and typedefs
//

//
// static data member definitions
//

double L1ExtraParticlesProd::muonMassGeV_ = 0.105658369 ; // PDG06

//
// constructors and destructor
//
L1ExtraParticlesProd::L1ExtraParticlesProd(const edm::ParameterSet& iConfig)
   : produceMuonParticles_( iConfig.getParameter< bool >(
      "produceMuonParticles" ) ),
     muonSource_( iConfig.getParameter< edm::InputTag >(
	"muonSource" ) ),
     produceCaloParticles_( iConfig.getParameter< bool >(
	"produceCaloParticles" ) ),
     isoEmSource_( iConfig.getParameter< edm::InputTag >(
	"isolatedEmSource" ) ),
     nonIsoEmSource_( iConfig.getParameter< edm::InputTag >(
	"nonIsolatedEmSource" ) ),
     cenJetSource_( iConfig.getParameter< edm::InputTag >(
	"centralJetSource" ) ),
     forJetSource_( iConfig.getParameter< edm::InputTag >(
	"forwardJetSource" ) ),
     tauJetSource_( iConfig.getParameter< edm::InputTag >(
	"tauJetSource" ) ),
     etTotSource_( iConfig.getParameter< edm::InputTag >(
	"etTotalSource" ) ),
     etHadSource_( iConfig.getParameter< edm::InputTag >(
	"etHadSource" ) ),
     etMissSource_( iConfig.getParameter< edm::InputTag >(
	"etMissSource" ) ),
     htMissSource_( iConfig.getParameter< edm::InputTag >(
	"htMissSource" ) ),
     hfRingEtSumsSource_( iConfig.getParameter< edm::InputTag >(
	"hfRingEtSumsSource" ) ),
     hfRingBitCountsSource_( iConfig.getParameter< edm::InputTag >(
	"hfRingBitCountsSource" ) ),
     centralBxOnly_( iConfig.getParameter< bool >(
	"centralBxOnly" ) )
{
   using namespace l1extra ;

   //register your products
   produces< L1EmParticleCollection >( "Isolated" ) ;
   produces< L1EmParticleCollection >( "NonIsolated" ) ;
   produces< L1JetParticleCollection >( "Central" ) ;
   produces< L1JetParticleCollection >( "Forward" ) ;
   produces< L1JetParticleCollection >( "Tau" ) ;
   produces< L1MuonParticleCollection >() ;
   produces< L1EtMissParticleCollection >( "MET" ) ;
   produces< L1EtMissParticleCollection >( "MHT" ) ;
   produces< L1HFRingsCollection >() ;

   //now do what ever other initialization is needed
}


L1ExtraParticlesProd::~L1ExtraParticlesProd()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1ExtraParticlesProd::produce( edm::Event& iEvent,
			       const edm::EventSetup& iSetup)
{
   using namespace edm ;
   using namespace l1extra ;
   using namespace std ;

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~~ Muons ~~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   if( produceMuonParticles_ )
   {
      ESHandle< L1MuTriggerScales > muScales ;
      iSetup.get< L1MuTriggerScalesRcd >().get( muScales ) ;

      ESHandle< L1MuTriggerPtScale > muPtScale ;
      iSetup.get< L1MuTriggerPtScaleRcd >().get( muPtScale ) ;

      Handle< L1MuGMTReadoutCollection > hwMuCollection ;
      iEvent.getByLabel( muonSource_, hwMuCollection ) ;

      vector< L1MuGMTExtendedCand > hwMuCands ;

      if( centralBxOnly_ )
      {
	 // Get GMT candidates from central bunch crossing only
	 hwMuCands = hwMuCollection->getRecord().getGMTCands() ;
      }
      else
      {
	 // Get GMT candidates from all bunch crossings
	 vector< L1MuGMTReadoutRecord > records = hwMuCollection->getRecords();
	 vector< L1MuGMTReadoutRecord >::const_iterator rItr = records.begin();
	 vector< L1MuGMTReadoutRecord >::const_iterator rEnd = records.end();

	 for( ; rItr != rEnd ; ++rItr )
	 {
	    vector< L1MuGMTExtendedCand > tmpCands = rItr->getGMTCands() ;

	    hwMuCands.insert( hwMuCands.end(),
			      tmpCands.begin(),
			      tmpCands.end() ) ;
	 }
      }

      auto_ptr< L1MuonParticleCollection > muColl(
	 new L1MuonParticleCollection );

//       cout << "HW muons" << endl ;
      vector< L1MuGMTExtendedCand >::const_iterator muItr = hwMuCands.begin() ;
      vector< L1MuGMTExtendedCand >::const_iterator muEnd = hwMuCands.end() ;
      for( int i = 0 ; muItr != muEnd ; ++muItr, ++i )
      {
// 	 cout << "#" << i
// 	      << " name " << muItr->name()
// 	      << " empty " << muItr->empty()
// 	      << " pt " << muItr->ptIndex()
// 	      << " eta " << muItr->etaIndex()
// 	      << " phi " << muItr->phiIndex()
// 	      << " iso " << muItr->isol()
// 	      << " mip " << muItr->mip()
// 	      << " bx " << muItr->bx()
// 	      << endl ;

	 if( !muItr->empty() )
	 {
	    // keep x and y components non-zero and protect against roundoff.
	    double pt =
	      muPtScale->getPtScale()->getLowEdge( muItr->ptIndex() ) + 1.e-6 ;
	    // muScales->getPtScale()->getLowEdge( muItr->ptIndex() ) + 1.e-6 ;

// 	    cout << "L1Extra pt " << pt << endl ;

	    double eta =
	       muScales->getGMTEtaScale()->getCenter( muItr->etaIndex() ) ;

	    double phi =
	       muScales->getPhiScale()->getLowEdge( muItr->phiIndex() ) ;

	    math::PtEtaPhiMLorentzVector p4( pt,
					     eta,
					     phi,
					     muonMassGeV_ ) ;

	    muColl->push_back(
	       L1MuonParticle( muItr->charge(),
			       p4,
			       *muItr,
			       muItr->bx() )
 	       ) ;
	 }
      }

      OrphanHandle< L1MuonParticleCollection > muHandle =
	 iEvent.put( muColl );
   }
   

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~~ Calorimeter ~~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   if( produceCaloParticles_ )
   {
      // ~~~~~~~~~~~~~~~~~~~~ Geometry ~~~~~~~~~~~~~~~~~~~~

      ESHandle< L1CaloGeometry > caloGeomESH ;
      iSetup.get< L1CaloGeometryRecord >().get( caloGeomESH ) ;
      const L1CaloGeometry* caloGeom = &( *caloGeomESH ) ;

      // ~~~~~~~~~~~~~~~~~~~~ EM ~~~~~~~~~~~~~~~~~~~~

      ESHandle< L1CaloEtScale > emScale ;
      iSetup.get< L1EmEtScaleRcd >().get( emScale ) ;

      // Isolated EM
      Handle< L1GctEmCandCollection > hwIsoEmCands ;
      iEvent.getByLabel( isoEmSource_, hwIsoEmCands ) ;

      auto_ptr< L1EmParticleCollection > isoEmColl(
	 new L1EmParticleCollection );

//       cout << "HW iso EM" << endl ;

      L1GctEmCandCollection::const_iterator emItr = hwIsoEmCands->begin() ;
      L1GctEmCandCollection::const_iterator emEnd = hwIsoEmCands->end() ;
      for( int i = 0 ; emItr != emEnd ; ++emItr, ++i )
      {
// 	 cout << "#" << i
// 	      << " name " << emItr->name()
// 	      << " empty " << emItr->empty()
// 	      << " rank " << emItr->rank()
// 	      << " eta " << emItr->etaIndex()
// 	      << " sign " << emItr->etaSign()
// 	      << " phi " << emItr->phiIndex()
// 	      << " iso " << emItr->isolated()
// 	      << " bx " << emItr->bx()
// 	      << endl ;

	 if( !emItr->empty() &&
	     ( !centralBxOnly_ || emItr->bx() == 0 ) )
	 {
	    double et = emScale->et( emItr->rank() ) ;

// 	    cout << "L1Extra et " << et << endl ;

	    isoEmColl->push_back(
	       L1EmParticle( gctLorentzVector( et, *emItr, caloGeom, true ),
			     Ref< L1GctEmCandCollection >( hwIsoEmCands,
							   i ),
			     emItr->bx() ) ) ;
	 }
      }

      OrphanHandle< L1EmParticleCollection > isoEmHandle =
	 iEvent.put( isoEmColl, "Isolated" ) ;


      // Non-isolated EM
      Handle< L1GctEmCandCollection > hwNonIsoEmCands ;
      iEvent.getByLabel( nonIsoEmSource_, hwNonIsoEmCands ) ;

      auto_ptr< L1EmParticleCollection > nonIsoEmColl(
	 new L1EmParticleCollection );

//       cout << "HW non-iso EM" << endl ;
      emItr = hwNonIsoEmCands->begin() ;
      emEnd = hwNonIsoEmCands->end() ;
      for( int i = 0 ; emItr != emEnd ; ++emItr, ++i )
      {
// 	 cout << "#" << i
// 	      << " name " << emItr->name()
// 	      << " empty " << emItr->empty()
// 	      << " rank " << emItr->rank()
// 	      << " eta " << emItr->etaIndex()
// 	      << " sign " << emItr->etaSign()
// 	      << " phi " << emItr->phiIndex()
// 	      << " iso " << emItr->isolated()
// 	      << " bx " << emItr->bx()
// 	      << endl ;

	 if( !emItr->empty() &&
	     ( !centralBxOnly_ || emItr->bx() == 0 ) )
	 {
	    double et = emScale->et( emItr->rank() ) ;

// 	    cout << "L1Extra et " << et << endl ;

	    nonIsoEmColl->push_back(
	       L1EmParticle( gctLorentzVector( et, *emItr, caloGeom, true ),
			     Ref< L1GctEmCandCollection >( hwNonIsoEmCands,
							   i ),
			     emItr->bx() ) );
	 }
      }

      OrphanHandle< L1EmParticleCollection > nonIsoEmHandle =
	 iEvent.put( nonIsoEmColl, "NonIsolated" ) ;


      // ~~~~~~~~~~~~~~~~~~~~ Jets ~~~~~~~~~~~~~~~~~~~~

      ESHandle< L1CaloEtScale > jetScale ;
      iSetup.get< L1JetEtScaleRcd >().get( jetScale ) ;

      // Central jets.
      Handle< L1GctJetCandCollection > hwCenJetCands ;
      iEvent.getByLabel( cenJetSource_, hwCenJetCands ) ;

      auto_ptr< L1JetParticleCollection > cenJetColl(
	 new L1JetParticleCollection );

//       cout << "HW central jets" << endl ;
      L1GctJetCandCollection::const_iterator jetItr = hwCenJetCands->begin() ;
      L1GctJetCandCollection::const_iterator jetEnd = hwCenJetCands->end() ;
      for( int i = 0 ; jetItr != jetEnd ; ++jetItr, ++i )
      {
// 	 cout << "#" << i
// 	      << " name " << jetItr->name()
// 	      << " empty " << jetItr->empty()
// 	      << " rank " << jetItr->rank()
// 	      << " eta " << jetItr->etaIndex()
// 	      << " sign " << jetItr->etaSign()
// 	      << " phi " << jetItr->phiIndex()
// 	      << " cen " << jetItr->isCentral()
// 	      << " for " << jetItr->isForward()
// 	      << " tau " << jetItr->isTau()
// 	      << " bx " << jetItr->bx()
// 	      << endl ;

	 if( !jetItr->empty() &&
	     ( !centralBxOnly_ || jetItr->bx() == 0 ) )
	 {
	    double et = jetScale->et( jetItr->rank() ) ;

// 	    cout << "L1Extra et " << et << endl ;

	    cenJetColl->push_back(
	       L1JetParticle( gctLorentzVector( et, *jetItr, caloGeom, true ),
			      Ref< L1GctJetCandCollection >( hwCenJetCands,
							     i ),
			      jetItr->bx() ) ) ;
	 }
      }

      OrphanHandle< L1JetParticleCollection > cenJetHandle =
	 iEvent.put( cenJetColl, "Central" ) ;


      // Forward jets.
      Handle< L1GctJetCandCollection > hwForJetCands ;
      iEvent.getByLabel( forJetSource_, hwForJetCands ) ;

      auto_ptr< L1JetParticleCollection > forJetColl(
	 new L1JetParticleCollection );

//       cout << "HW forward jets" << endl ;
      jetItr = hwForJetCands->begin() ;
      jetEnd = hwForJetCands->end() ;
      for( int i = 0 ; jetItr != jetEnd ; ++jetItr, ++i )
      {
// 	 cout << "#" << i
// 	      << " name " << jetItr->name()
// 	      << " empty " << jetItr->empty()
// 	      << " rank " << jetItr->rank()
// 	      << " eta " << jetItr->etaIndex()
// 	      << " sign " << jetItr->etaSign()
// 	      << " phi " << jetItr->phiIndex()
// 	      << " cen " << jetItr->isCentral()
// 	      << " for " << jetItr->isForward()
// 	      << " tau " << jetItr->isTau()
// 	      << " bx " << jetItr->bx()
// 	      << endl ;

	 if( !jetItr->empty() &&
	     ( !centralBxOnly_ || jetItr->bx() == 0 ) )
	 {
	    double et = jetScale->et( jetItr->rank() ) ;

// 	    cout << "L1Extra et " << et << endl ;

	    forJetColl->push_back(
	       L1JetParticle( gctLorentzVector( et, *jetItr, caloGeom, false ),
			      Ref< L1GctJetCandCollection >( hwForJetCands,
							     i ),
			      jetItr->bx() ) ) ;
	 }
      }

      OrphanHandle< L1JetParticleCollection > forJetHandle =
	 iEvent.put( forJetColl, "Forward" ) ;


      // Tau jets.
//       cout << "HW tau jets" << endl ;
      Handle< L1GctJetCandCollection > hwTauJetCands ;
      iEvent.getByLabel( tauJetSource_, hwTauJetCands ) ;

      auto_ptr< L1JetParticleCollection > tauJetColl(
	 new L1JetParticleCollection );

      jetItr = hwTauJetCands->begin() ;
      jetEnd = hwTauJetCands->end() ;
      for( int i = 0 ; jetItr != jetEnd ; ++jetItr, ++i )
      {
// 	 cout << "#" << i
// 	      << " name " << jetItr->name()
// 	      << " empty " << jetItr->empty()
// 	      << " rank " << jetItr->rank()
// 	      << " eta " << jetItr->etaIndex()
// 	      << " sign " << jetItr->etaSign()
// 	      << " phi " << jetItr->phiIndex()
// 	      << " cen " << jetItr->isCentral()
// 	      << " for " << jetItr->isForward()
// 	      << " tau " << jetItr->isTau()
// 	      << " bx " << jetItr->bx()
// 	      << endl ;

	 if( !jetItr->empty() &&
	     ( !centralBxOnly_ || jetItr->bx() == 0 ) )
	 {
	    double et = jetScale->et( jetItr->rank() ) ;

// 	    cout << "L1Extra et " << et << endl ;

	    tauJetColl->push_back(
	       L1JetParticle( gctLorentzVector( et, *jetItr, caloGeom, true ),
			      Ref< L1GctJetCandCollection >( hwTauJetCands,
							     i ),
			      jetItr->bx() ) ) ;
	 }
      }

      OrphanHandle< L1JetParticleCollection > tauJetHandle =
	 iEvent.put( tauJetColl, "Tau" ) ;

      // ~~~~~~~~~~~~~~~~~~~~ ET Sums ~~~~~~~~~~~~~~~~~~~~

      double etSumLSB = jetScale->linearLsb() ;

      Handle< L1GctEtTotalCollection > hwEtTotColl ;
      iEvent.getByLabel( etTotSource_, hwEtTotColl ) ;

      Handle< L1GctEtMissCollection > hwEtMissColl ;
      iEvent.getByLabel( etMissSource_, hwEtMissColl ) ;

      auto_ptr< L1EtMissParticleCollection > etMissColl(
	 new L1EtMissParticleCollection );

      // Collate energy sums by bx
      L1GctEtTotalCollection::const_iterator hwEtTotItr =
	 hwEtTotColl->begin() ;
      L1GctEtTotalCollection::const_iterator hwEtTotEnd =
	 hwEtTotColl->end() ;

      int iTot = 0 ;
      for( ; hwEtTotItr != hwEtTotEnd ; ++hwEtTotItr, ++iTot )
      {
	 int bx = hwEtTotItr->bx() ;

	 if( !centralBxOnly_ || bx == 0 )
	 {
	   L1GctEtMissCollection::const_iterator hwEtMissItr =
	     hwEtMissColl->begin() ;
	   L1GctEtMissCollection::const_iterator hwEtMissEnd =
	     hwEtMissColl->end() ;

	   int iMiss = 0 ;
	   for( ; hwEtMissItr != hwEtMissEnd ; ++hwEtMissItr, ++iMiss )
	     {
	       if( hwEtMissItr->bx() == bx )
		 {
		   break ;
		 }
	     }

	   // If a L1GctEtMiss with the right bx is not found, itr == end.
	   if( hwEtMissItr != hwEtMissEnd )
	     {
	       // Construct L1EtMissParticle only if both energy
	       // sums are present.

	       // ET bin low edge
	       double etTot =
		 ( hwEtTotItr->overFlow() ?
		   ( double ) L1GctEtTotal::kEtTotalMaxValue :
		   ( double ) hwEtTotItr->et() ) * etSumLSB + 1.e-6 ;
	       double etMiss =
		 ( hwEtMissItr->overFlow() ?
		   ( double ) L1GctEtMiss::kEtMissMaxValue :
		   ( double ) hwEtMissItr->et() ) * etSumLSB + 1.e-6 ;
	       // keep x and y components non-zero and
	       // protect against roundoff.

	       double phi =
		 caloGeom->etSumPhiBinCenter( hwEtMissItr->phi() ) ;

	       cout << "HW ET Sums " << endl
		    << "MET: phi " << hwEtMissItr->phi() << " = " << phi
		    << " et " << hwEtMissItr->et() << " = " << etMiss
		    << " EtTot " << hwEtTotItr->et() << " = " << etTot
		    << " bx " << bx
		    << endl ;

	       math::PtEtaPhiMLorentzVector p4( etMiss,
						0.,
						phi,
						0. ) ;

	       etMissColl->push_back(
		 L1EtMissParticle(
		    p4,
		    L1EtMissParticle::kMET,
		    etTot,
		    Ref< L1GctEtMissCollection >( hwEtMissColl, iMiss ),
		    Ref< L1GctEtTotalCollection >( hwEtTotColl, iTot ),
		    Ref< L1GctHtMissCollection >(),
		    Ref< L1GctEtHadCollection >(),
		    bx
		    ) ) ;
	     }
	 }
      }

      OrphanHandle< L1EtMissParticleCollection > etMissCollHandle =
	 iEvent.put( etMissColl, "MET" ) ;

      // ~~~~~~~~~~~~~~~~~~~~ HT Sums ~~~~~~~~~~~~~~~~~~~~

      Handle< L1GctEtHadCollection > hwEtHadColl ;
      iEvent.getByLabel( etHadSource_, hwEtHadColl ) ;

      Handle< L1GctHtMissCollection > hwHtMissColl ;
      iEvent.getByLabel( htMissSource_, hwHtMissColl ) ;

//       ESHandle< L1GctJetFinderParams > jetFinderParams ;
//       iSetup.get< L1GctJetFinderParamsRcd >().get( jetFinderParams ) ;
//       double htSumLSB = jetFinderParams->getHtLsbGeV();
      double htSumLSB = 1. ;

      auto_ptr< L1EtMissParticleCollection > htMissColl(
	 new L1EtMissParticleCollection );

      L1GctEtHadCollection::const_iterator hwEtHadItr =
	hwEtHadColl->begin() ;
      L1GctEtHadCollection::const_iterator hwEtHadEnd =
	hwEtHadColl->end() ;

      int iHad = 0 ;
      for( ; hwEtHadItr != hwEtHadEnd ; ++hwEtHadItr, ++iHad )
      {
	 int bx = hwEtHadItr->bx() ;

	 if( !centralBxOnly_ || bx == 0 )
	 {
	   L1GctHtMissCollection::const_iterator hwHtMissItr =
	     hwHtMissColl->begin() ;
	   L1GctHtMissCollection::const_iterator hwHtMissEnd =
	     hwHtMissColl->end() ;

	   int iMiss = 0 ;
	   for( ; hwHtMissItr != hwHtMissEnd ; ++hwHtMissItr, ++iMiss )
	     {
	       if( hwHtMissItr->bx() == bx )
		 {
		   break ;
		 }
	     }

	   // If a L1GctHtMiss with the right bx is not found, itr == end.
	   if( hwHtMissItr != hwHtMissEnd )
	     {
	       // Construct L1EtMissParticle only if both energy
	       // sums are present.

	       // HT bin low edge
	       double htTot =
		 ( hwEtHadItr->overFlow() ?
		   ( double ) L1GctEtHad::kEtHadMaxValue :
		   ( double ) hwEtHadItr->et() ) * htSumLSB + 1.e-6 ;
	       double htMiss =
		 ( hwHtMissItr->overFlow() ?
		   ( double ) L1GctHtMiss::kHtMissOFlowBit - 1. :
		   ( double ) hwHtMissItr->et() ) * htSumLSB + 1.e-6 ;
	       // keep x and y components non-zero and
	       // protect against roundoff.

	       double phi =
		 caloGeom->etSumPhiBinCenter( hwHtMissItr->phi() ) ;

	       cout << "HW HT Sums " << endl
		    << "MHT: phi " << hwHtMissItr->phi() << " = " << phi
		    << " ht " << hwHtMissItr->et() << " = " << htMiss
		    << " HtTot " << hwEtHadItr->et() << " = " << htTot
		    << " bx " << bx
		    << endl ;

	       math::PtEtaPhiMLorentzVector p4( htMiss,
						0.,
						phi,
						0. ) ;

	       htMissColl->push_back(
		 L1EtMissParticle(
		   p4,
		   L1EtMissParticle::kMHT,
		   htTot,
		   Ref< L1GctEtMissCollection >(),
		   Ref< L1GctEtTotalCollection >(),
		   Ref< L1GctHtMissCollection >( hwHtMissColl, iMiss ),
		   Ref< L1GctEtHadCollection >( hwEtHadColl, iHad ),
		   bx
		   ) ) ;
	     }
	 }
      }

      OrphanHandle< L1EtMissParticleCollection > htMissCollHandle =
	 iEvent.put( htMissColl, "MHT" ) ;

      // ~~~~~~~~~~~~~~~~~~~~ HF Rings ~~~~~~~~~~~~~~~~~~~~

      Handle< L1GctHFRingEtSumsCollection > hwHFEtSumsColl ;
      iEvent.getByLabel( hfRingEtSumsSource_, hwHFEtSumsColl ) ;

      Handle< L1GctHFBitCountsCollection > hwHFBitCountsColl ;
      iEvent.getByLabel( hfRingBitCountsSource_, hwHFBitCountsColl ) ;

      auto_ptr< L1HFRingsCollection > hfRingsColl( new L1HFRingsCollection );

      L1GctHFRingEtSumsCollection::const_iterator hwHFEtSumsItr =
	hwHFEtSumsColl->begin() ;
      L1GctHFRingEtSumsCollection::const_iterator hwHFEtSumsEnd =
	hwHFEtSumsColl->end() ;

      int iEtSums = 0 ;
      for( ; hwHFEtSumsItr != hwHFEtSumsEnd ; ++hwHFEtSumsItr, ++iEtSums )
      {
	 int bx = hwHFEtSumsItr->bx() ;

	 if( !centralBxOnly_ || bx == 0 )
	 {
	   L1GctHFBitCountsCollection::const_iterator hwHFBitCountsItr =
	     hwHFBitCountsColl->begin() ;
	   L1GctHFBitCountsCollection::const_iterator hwHFBitCountsEnd =
	     hwHFBitCountsColl->end() ;

	   int iBitCounts = 0 ;
	   for( ; hwHFBitCountsItr != hwHFBitCountsEnd ;
		++hwHFBitCountsItr, ++iBitCounts )
	     {
	       if( hwHFBitCountsItr->bx() == bx )
		 {
		   break ;
		 }
	     }

	   // If a L1GctHFBitCounts with the right bx is not found, itr == end.
	   if( hwHFBitCountsItr != hwHFBitCountsEnd )
	     {
	       // Construct L1HFRings only if both HW objects are present.

	       // cout << "HF Rings " << endl

	       // HF Et sums
	       double etSums[ L1HFRings::kNumRings ] ;
	       etSums[ L1HFRings::kRing1PosEta ] = 0. ;
	       etSums[ L1HFRings::kRing1NegEta ] = 0. ;
	       etSums[ L1HFRings::kRing2PosEta ] = 0. ;
	       etSums[ L1HFRings::kRing2NegEta ] = 0. ;

// 	       double etHad =
// 		 ( hwEtHadItr->overFlow() ?
// 		   ( double ) L1GctEtHad::kEtHadMaxValue :
// 		   ( double ) hwEtHadItr->et() ) * htSumLSB + 1.e-6 ;
// 	       double htMiss =
// 		 ( hwHtMissItr->overFlow() ?
// 		   ( double ) L1GctHtMiss::kHtMissOFlowBit - 1. :
// 		   ( double ) hwHtMissItr->et() ) * htSumLSB + 1.e-6 ;
// 	       // protect against roundoff.

	       // HF bit counts
	       int bitCounts[ L1HFRings::kNumRings ] ;
	       bitCounts[ L1HFRings::kRing1PosEta ] =
		 hwHFBitCountsItr->bitCount( 0 ) ;
	       bitCounts[ L1HFRings::kRing1NegEta ] =
		 hwHFBitCountsItr->bitCount( 1 ) ;
	       bitCounts[ L1HFRings::kRing2PosEta ] =
		 hwHFBitCountsItr->bitCount( 2 ) ;
	       bitCounts[ L1HFRings::kRing2NegEta ] =
		 hwHFBitCountsItr->bitCount( 3 ) ;

	       hfRingsColl->push_back(
		 L1HFRings( etSums,
			    bitCounts,
			    Ref< L1GctHFRingEtSumsCollection >( hwHFEtSumsColl,
								iEtSums ),
			    Ref< L1GctHFBitCountsCollection >( hwHFBitCountsColl,
							       iBitCounts ),
			    bx
			    ) ) ;
	     }
	 }
      }

      OrphanHandle< L1HFRingsCollection > hfRingsCollHandle =
	 iEvent.put( hfRingsColl ) ;
   }
}

//math::XYZTLorentzVector
math::PtEtaPhiMLorentzVector
L1ExtraParticlesProd::gctLorentzVector( const double& et,
					const L1GctCand& cand,
					const L1CaloGeometry* geom,
					bool central )
{
   // To keep x and y components non-zero.
   double etCorr = et + 1.e-6 ; // protect against roundoff, not only for et=0

   double eta = geom->etaBinCenter( cand.etaIndex(), central ) ;

//    double tanThOver2 = exp( -eta ) ;
//    double ez = etCorr * ( 1. - tanThOver2 * tanThOver2 ) / ( 2. * tanThOver2 );
//    double e  = etCorr * ( 1. + tanThOver2 * tanThOver2 ) / ( 2. * tanThOver2 );

   double phi = geom->emJetPhiBinCenter( cand.phiIndex() ) ;

//    return math::XYZTLorentzVector( etCorr * cos( phi ),
// 				   etCorr * sin( phi ),
// 				   ez,
// 				   e ) ;
   return math::PtEtaPhiMLorentzVector( etCorr,
					eta,
					phi,
					0. ) ;
}     

// ------------ method called once each job just before starting event loop  ------------
void 
L1ExtraParticlesProd::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1ExtraParticlesProd::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1ExtraParticlesProd);
