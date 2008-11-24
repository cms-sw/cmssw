// -*- C++ -*-
//
// Package:    L1ExtraTestAnalyzer
// Class:      L1ExtraTestAnalyzer
// 
/**\class L1ExtraTestAnalyzer \file L1ExtraTestAnalyzer.cc L1TriggerOffline/L1ExtraTestAnalyzer/src/L1ExtraTestAnalyzer.cc \author Werner Sun

   Description: simple analyzer to print out L1Extra object information.
*/
//
// Original Author:  Werner Sun
//         Created:  Fri Jul 28 14:22:31 EDT 2006
// $Id: L1ExtraTestAnalyzer.cc,v 1.5 2007/12/18 03:32:44 wsun Exp $
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TH1.h"
#include "TFile.h"

using namespace std ;

//
// class decleration
//

class L1ExtraTestAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L1ExtraTestAnalyzer(const edm::ParameterSet&);
      ~L1ExtraTestAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

      edm::InputTag isoEmSource_ ;
      edm::InputTag nonIsoEmSource_ ;
      edm::InputTag cenJetSource_ ;
      edm::InputTag forJetSource_ ;
      edm::InputTag tauJetSource_ ;
      edm::InputTag muonSource_ ;
      edm::InputTag etMissSource_ ;
      edm::InputTag gtReadoutSource_ ;
      edm::InputTag particleMapSource_ ;

      TFile file_ ;
      TH1F hist_ ;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1ExtraTestAnalyzer::L1ExtraTestAnalyzer(const edm::ParameterSet& iConfig)
   : isoEmSource_( iConfig.getParameter< edm::InputTag >(
      "isolatedEmSource" ) ),
     nonIsoEmSource_( iConfig.getParameter< edm::InputTag >(
      "nonIsolatedEmSource" ) ),
     cenJetSource_( iConfig.getParameter< edm::InputTag >(
      "centralJetSource" ) ),
     forJetSource_( iConfig.getParameter< edm::InputTag >(
      "forwardJetSource" ) ),
     tauJetSource_( iConfig.getParameter< edm::InputTag >(
      "tauJetSource" ) ),
     muonSource_( iConfig.getParameter< edm::InputTag >(
      "muonSource" ) ),
     etMissSource_( iConfig.getParameter< edm::InputTag >(
      "etMissSource" ) ),
     gtReadoutSource_( iConfig.getParameter< edm::InputTag >(
      "gtReadoutSource" ) ),
     particleMapSource_( iConfig.getParameter< edm::InputTag >(
      "particleMapSource" ) ),
     file_( "l1extra.root", "RECREATE" ),
     hist_( "triggers", "Triggers",
	    2*l1extra::L1ParticleMap::kNumOfL1TriggerTypes + 1,
	    -0.75,
	    l1extra::L1ParticleMap::kNumOfL1TriggerTypes + 0.5 - 0.75 )
{
   //now do what ever initialization is needed

}


L1ExtraTestAnalyzer::~L1ExtraTestAnalyzer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

   file_.cd() ;
   hist_.Write() ;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1ExtraTestAnalyzer::analyze(const edm::Event& iEvent,
			     const edm::EventSetup& iSetup)
{
   using namespace edm ;
   using namespace l1extra ;

   static int iev = 0 ;
   cout << "EVENT " << ++iev << endl ;

   cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl ;

   // Isolated EM particles
   Handle< L1EmParticleCollection > isoEmColl ;
   iEvent.getByLabel( isoEmSource_, isoEmColl ) ;
   cout << "Number of isolated EM " << isoEmColl->size() << endl ;

   for( L1EmParticleCollection::const_iterator emItr = isoEmColl->begin() ;
        emItr != isoEmColl->end() ;
        ++emItr )
   {
      cout << "  p4 (" << emItr->px()
	   << ", " << emItr->py()
	   << ", " << emItr->pz()
	   << ", " << emItr->energy()
	   << ") et " << emItr->et()
	   << " eta " << emItr->eta()
	   << " phi " << emItr->phi()
	   << endl ;
   }

   // Non-isolated EM particles
   Handle< L1EmParticleCollection > nonIsoEmColl ;
   iEvent.getByLabel( nonIsoEmSource_, nonIsoEmColl ) ;
   cout << "Number of non-isolated EM " << nonIsoEmColl->size() << endl ;

   for( L1EmParticleCollection::const_iterator emItr = nonIsoEmColl->begin() ;
        emItr != nonIsoEmColl->end() ;
        ++emItr )
   {
      cout << "  p4 (" << emItr->px()
	   << ", " << emItr->py()
	   << ", " << emItr->pz()
	   << ", " << emItr->energy()
	   << ") et " << emItr->et()
	   << " eta " << emItr->eta()
	   << " phi " << emItr->phi()
	   << endl ;
   }

   // Jet particles
   Handle< L1JetParticleCollection > cenJetColl ;
   iEvent.getByLabel( cenJetSource_, cenJetColl ) ;
   cout << "Number of central jets " << cenJetColl->size() << endl ;

   for( L1JetParticleCollection::const_iterator jetItr = cenJetColl->begin() ;
        jetItr != cenJetColl->end() ;
        ++jetItr )
   {
      cout << "  p4 (" << jetItr->px()
	   << ", " << jetItr->py()
	   << ", " << jetItr->pz()
	   << ", " << jetItr->energy()
	   << ") et " << jetItr->et()
	   << " eta " << jetItr->eta()
	   << " phi " << jetItr->phi()
	   << endl ;
   }

   Handle< L1JetParticleCollection > forJetColl ;
   iEvent.getByLabel( forJetSource_, forJetColl ) ;
   cout << "Number of forward jets " << forJetColl->size() << endl ;

   for( L1JetParticleCollection::const_iterator jetItr = forJetColl->begin() ;
        jetItr != forJetColl->end() ;
        ++jetItr )
   {
      cout << "  p4 (" << jetItr->px()
	   << ", " << jetItr->py()
	   << ", " << jetItr->pz()
	   << ", " << jetItr->energy()
	   << ") et " << jetItr->et()
	   << " eta " << jetItr->eta()
	   << " phi " << jetItr->phi()
	   << endl ;
   }

   Handle< L1JetParticleCollection > tauColl ;
   iEvent.getByLabel( tauJetSource_, tauColl ) ;
   cout << "Number of tau jets " << tauColl->size() << endl ;

   for( L1JetParticleCollection::const_iterator tauItr = tauColl->begin() ;
        tauItr != tauColl->end() ;
        ++tauItr )
   {
      cout << "  p4 (" << tauItr->px()
	   << ", " << tauItr->py()
	   << ", " << tauItr->pz()
	   << ", " << tauItr->energy()
	   << ") et " << tauItr->et()
	   << " eta " << tauItr->eta()
	   << " phi " << tauItr->phi()
	   << endl ;
   }

   // Muon particles
   Handle< L1MuonParticleCollection > muColl ;
   iEvent.getByLabel( muonSource_, muColl ) ;
   cout << "Number of muons " << muColl->size() << endl ;

   for( L1MuonParticleCollection::const_iterator muItr = muColl->begin() ;
        muItr != muColl->end() ;
        ++muItr )
   {
      cout << "  q " << muItr->charge()
	   << " p4 (" << muItr->px()
	   << ", " << muItr->py()
	   << ", " << muItr->pz()
	   << ", " << muItr->energy()
	   << ") et " << muItr->et()
	   << " eta " << muItr->eta() << endl
           << "    phi " << muItr->phi()
           << " iso " << muItr->isIsolated()
	   << " mip " << muItr->isMip()
	   << " fwd " << muItr->isForward()
	   << " rpc " << muItr->isRPC()
	   << endl ;
   }

   // MET particle
//    Handle< L1EtMissParticle > etMiss ;
//    iEvent.getByLabel( etMissSource_, etMiss ) ;
//    cout << "MET (" << etMiss->px()
// 	<< ", " << etMiss->py()
// 	<< ", " << etMiss->pz()
// 	<< ", " << etMiss->energy()
//         << ") phi " << etMiss->phi()
//         << " EtTot " << etMiss->etTotal()
// 	<< " EtHad " << etMiss->etHad()
// 	<< endl ;

   Handle< L1EtMissParticleCollection > etMissColl ;
   iEvent.getByLabel( etMissSource_, etMissColl ) ;
   cout << "MET Coll (" << etMissColl->begin()->px()
	<< ", " << etMissColl->begin()->py()
	<< ", " << etMissColl->begin()->pz()
	<< ", " << etMissColl->begin()->energy()
        << ") phi " << etMissColl->begin()->phi()
        << " EtTot " << etMissColl->begin()->etTotal()
	<< " EtHad " << etMissColl->begin()->etHad()
	<< endl ;


//    // L1GlobalTriggerReadoutRecord
//    Handle< L1GlobalTriggerReadoutRecord > gtRecord ;
//    iEvent.getByLabel( gtReadoutSource_, gtRecord ) ;
//    cout << "Global trigger decision " << gtRecord->decision() << endl ;


//    // L1ParticleMaps
//    Handle< L1ParticleMapCollection > mapColl ;
//    iEvent.getByLabel( particleMapSource_, mapColl ) ;
//    cout << "Number of particle maps " << mapColl->size() << endl ;

//    for( int imap = 0 ; imap < L1ParticleMap::kNumOfL1TriggerTypes ; ++imap )
//    {
//       const L1ParticleMap& map = ( *mapColl )[ imap ] ;

//       cout << "Trig #" << map.triggerType()
// 	   << " name " << map.triggerName()
// 	   << " decision " << map.triggerDecision()
// 	   << " #objs " << map.numOfObjects() ;
//       cout << " types" ;
//       for( int i = 0 ; i < map.numOfObjects() ; ++i )
//       {
// 	 cout << " " << map.objectTypes()[ i ] ;
//       }
//       cout << endl ;

//       hist_.Fill( imap, ( map.triggerDecision() ? 1. : 0. ) ) ;

//       if( map.triggerDecision() )
//       {
// 	 const L1EmParticleVectorRef& ems = map.emParticles() ;
// 	 cout << "  EM particles ET" ;
// 	 for( L1EmParticleVectorRef::const_iterator emItr = ems.begin() ;
// 	      emItr != ems.end() ; ++emItr )
// 	 {
// 	    cout << " " << ( *emItr )->et() ;
// 	 }
// 	 cout << endl ;

// 	 const L1JetParticleVectorRef& jets = map.jetParticles() ;
// 	 cout << "  Jet particles ET" ;
// 	 for( L1JetParticleVectorRef::const_iterator jetItr = jets.begin() ;
// 	      jetItr != jets.end() ; ++jetItr )
// 	 {
// 	    cout << " " << ( *jetItr )->et() ;
// 	 }
// 	 cout << endl ;

// 	 const L1MuonParticleVectorRef& muons = map.muonParticles() ;
// 	 cout << "  Muon particles ET" ;
// 	 for( L1MuonParticleVectorRef::const_iterator muonItr = muons.begin() ;
// 	      muonItr != muons.end() ; ++muonItr )
// 	 {
// 	    cout << " " << ( *muonItr )->et() ;
// 	 }
// 	 cout << endl ;

// 	 cout << "  Particle combinations" ;
// 	 const L1ParticleMap::L1IndexComboVector& combos = map.indexCombos() ;
// 	 for( L1ParticleMap::L1IndexComboVector::const_iterator comboItr =
// 		 combos.begin() ; comboItr != combos.end() ; ++comboItr )
// 	 {
// 	    cout << " {" ;
// 	    for( L1ParticleMap::L1IndexCombo::const_iterator indexItr =
// 		    comboItr->begin() ;
// 		 indexItr != comboItr->end() ; ++indexItr )
// 	    {
// 	       cout << *indexItr ;
// 	    }
// 	    cout << "}" ;
// 	 }

// 	 cout << endl ;
//       }
//    }

   cout << endl ;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1ExtraTestAnalyzer);
