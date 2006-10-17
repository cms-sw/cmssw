// -*- C++ -*-
//
// Package:    L1ExtraParticlesProd
// Class:      L1ExtraParticlesProd
// 
/**\class L1ExtraParticlesProd L1ExtraParticlesProd.cc src/L1ExtraParticlesProd/src/L1ExtraParticlesProd.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Werner Sun
//         Created:  Mon Oct  2 22:45:32 EDT 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticlesProd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

#include "L1Trigger/L1Scales/interface/L1CaloEtScale.h"
#include "L1Trigger/L1Scales/interface/L1EmEtScaleRcd.h"
#include "L1Trigger/L1Scales/interface/L1JetEtScaleRcd.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class decleration
//


//
// constants, enums and typedefs
//

//
// static data member definitions
//

double L1ExtraParticlesProd::muonMassGeV_ = 0.1134289264 ; // PDG04

double L1ExtraParticlesProd::gctPhiOffset_ = 0. ;
double L1ExtraParticlesProd::gctEmJetPhiBinWidth_ =
   2. * M_PI / L1ExtraParticlesProd::kNumberGctEmJetPhiBins ;
double L1ExtraParticlesProd::gctEtSumPhiBinWidth_ =
   2. * M_PI / L1ExtraParticlesProd::kNumberGctEtSumPhiBins ;

unsigned L1ExtraParticlesProd::gctEtaSignBitOffset_ = 8 ;

double L1ExtraParticlesProd::gctEtaBinBoundaries_[
   kNumberGctCentralEtaBinsPerHalf + kNumberGctForwardEtaBinsPerHalf + 1 ] = {
      0.0000,
      0.3480,
      0.6950,
      1.0440,
      1.3920,
      1.7400,
      2.1720,
      3.0000,
      3.5000,
      4.0000,
      4.5000,
      5.0000 } ;

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
	"etMissSource" ) )
{
   using namespace l1extra ;

   //register your products
   produces< L1EmParticleCollection >( "Isolated" ) ;
   produces< L1EmParticleCollection >( "NonIsolated" ) ;
   produces< L1JetParticleCollection >( "Central" ) ;
   produces< L1JetParticleCollection >( "Forward" ) ;
   produces< L1JetParticleCollection >( "Tau" ) ;
   produces< L1MuonParticleCollection >() ;
   produces< L1EtMissParticle >() ;

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
      L1MuTriggerScales muScales ;

      Handle< L1MuGMTReadoutCollection > hwMuCollection ;
      iEvent.getByLabel( muonSource_, hwMuCollection ) ;

      vector< L1MuGMTExtendedCand > hwMuCands =
	 hwMuCollection->getRecord().getGMTCands() ; // from default bx.

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
// 	      << endl ;

	 if( !muItr->empty() )
	 {
	    double pt = muScales.getPtScale()->getLowEdge( muItr->ptIndex() ) ;

	    double eta =
	       muScales.getGMTEtaScale()->getCenter( muItr->etaIndex() ) ;
	    double tanThOver2 = exp( -eta ) ;
	    double pz = pt * ( 1. - tanThOver2 * tanThOver2 ) /
	       ( 2. * tanThOver2 ) ;
	    double p  = pt * ( 1. + tanThOver2 * tanThOver2 ) /
	       ( 2. * tanThOver2 ) ;
	    double e = sqrt( p * p + muonMassGeV_ * muonMassGeV_ ) ;

	    double phi =
	       muScales.getPhiScale()->getLowEdge( muItr->phiIndex() ) ;

	    math::XYZTLorentzVector p4( pt * cos( phi ),
					pt * sin( phi ),
					pz,
					e ) ;

	    muColl->push_back(
	       L1MuonParticle( muItr->charge(),
			       p4,
			       muItr->isol(),
			       muItr->mip(),
			       muItr->isFwd(),
			       muItr->isRPC(),
			       muItr->detector() )
	       ) ;
// 			       Ref< vector< L1MuGMTCand> >( hwMuCollection,
// 							    i ) ) ) ;
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
// 	      << endl ;

	 if( !emItr->empty() )
	 {
	    double et = emScale->et( emItr->rank() ) ;

	    isoEmColl->push_back(
	       L1EmParticle( gctLorentzVector( et, *emItr, true ),
			     Ref< L1GctEmCandCollection >( hwIsoEmCands,
							   i ) ) ) ;
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
// 	      << endl ;

	 if( !emItr->empty() )
	 {
	    double et = emScale->et( emItr->rank() ) ;

	    nonIsoEmColl->push_back(
	       L1EmParticle( gctLorentzVector( et, *emItr, true ),
			     Ref< L1GctEmCandCollection >( hwNonIsoEmCands,
							   i ) ) );
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
// 	      << endl ;

	 if( !jetItr->empty() )
	 {
	    double et = jetScale->et( jetItr->rank() ) ;

	    cenJetColl->push_back(
	       L1JetParticle( gctLorentzVector( et, *jetItr, true ),
			      Ref< L1GctJetCandCollection >( hwCenJetCands,
							     i ) ) ) ;
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
// 	      << endl ;

	 if( !jetItr->empty() )
	 {
	    double et = jetScale->et( jetItr->rank() ) ;

	    forJetColl->push_back(
	       L1JetParticle( gctLorentzVector( et, *jetItr, false ),
			      Ref< L1GctJetCandCollection >( hwForJetCands,
							     i ) ) ) ;
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
// 	      << endl ;

	 if( !jetItr->empty() )
	 {
	    double et = jetScale->et( jetItr->rank() ) ;

	    tauJetColl->push_back(
	       L1JetParticle( gctLorentzVector( et, *jetItr, true ),
			      Ref< L1GctJetCandCollection >( hwTauJetCands,
							     i ) ) ) ;
	 }
      }

      OrphanHandle< L1JetParticleCollection > tauJetHandle =
	 iEvent.put( tauJetColl, "Tau" ) ;


      // ~~~~~~~~~~~~~~~~~~~~ Energy Sums ~~~~~~~~~~~~~~~~~~~~

      Handle< L1GctEtTotal > hwEtTot ;
      iEvent.getByLabel( etTotSource_, hwEtTot ) ;

      Handle< L1GctEtHad > hwEtHad ;
      iEvent.getByLabel( etHadSource_, hwEtHad ) ;

      Handle< L1GctEtMiss > hwEtMiss ;
      iEvent.getByLabel( etMissSource_, hwEtMiss ) ;

      //      double etSumLSB = jetScale->lsb() ;
      double etSumLSB = 1. ;

//       cout << "HW ET Sums " << endl
// 	   << "MET: phi " << hwEtMiss->phi() << " et " << hwEtMiss->et()
// 	   << " EtTot " << hwEtTot->et() << " EtHad " << hwEtHad->et()
// 	   << endl ;

      double etTot = ( ( double ) hwEtTot->et() + 0.5 ) * etSumLSB ;
      double etHad = ( ( double ) hwEtHad->et() + 0.5 ) * etSumLSB ;
      double etMiss = ( ( double ) hwEtMiss->et() + 0.5 ) * etSumLSB ;

      unsigned phiIndex = hwEtMiss->phi() ;
      double phi =
	 ( ( double ) phiIndex + 0.5 ) * gctEtSumPhiBinWidth_ + gctPhiOffset_ ;

      math::XYZTLorentzVector p4( etMiss * cos( phi ),
				  etMiss * sin( phi ),
				  0.,
				  etMiss ) ;

      auto_ptr< L1EtMissParticle > etMissParticle(
	 new L1EtMissParticle( p4,
			       etTot,
			       etHad,
			       RefProd< L1GctEtMiss >( hwEtMiss ),
			       RefProd< L1GctEtTotal >( hwEtTot ),
			       RefProd< L1GctEtHad >( hwEtHad )
	    ) ) ;

      OrphanHandle< L1EtMissParticle > etMissHandle =
	 iEvent.put( etMissParticle ) ;
   }
}

math::XYZTLorentzVector
L1ExtraParticlesProd::gctLorentzVector( const double& et,
					const L1GctCand& cand,
					bool central )
{
   // Central/tau jets and EM have etaIndex = 0-6 for eta = 0-3.
   // Forward jets have etaIndex = 0-3 for eta = 3-5.
   unsigned etaIndex = cand.etaIndex() ;
   double etaSign = 1. ;

   // Check sign BEFORE shifting forward jet bin index.
   if( etaIndex >= gctEtaSignBitOffset_ )
   {
      etaSign = -1. ;
      etaIndex -= gctEtaSignBitOffset_ ;
   }

   // Shift forward jet bin index AFTER checking sign bit.
   if( !central )
   {
      etaIndex += kNumberGctCentralEtaBinsPerHalf ;
   }

   double eta = 0.5 * etaSign *
      ( gctEtaBinBoundaries_[ etaIndex ] +
	gctEtaBinBoundaries_[ etaIndex + 1 ] ) ;

   double tanThOver2 = exp( -eta ) ;
   double ez = et * ( 1. - tanThOver2 * tanThOver2 ) / ( 2. * tanThOver2 ) ;
   double e  = et * ( 1. + tanThOver2 * tanThOver2 ) / ( 2. * tanThOver2 ) ;

   unsigned phiIndex = cand.phiIndex() ;
   double phi =
      ( ( double ) phiIndex + 0.5 ) * gctEmJetPhiBinWidth_ + gctPhiOffset_ ;

   return math::XYZTLorentzVector( et * cos( phi ),
				   et * sin( phi ),
				   ez,
				   e ) ;
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
//DEFINE_FWK_MODULE(L1ExtraParticlesProd)
