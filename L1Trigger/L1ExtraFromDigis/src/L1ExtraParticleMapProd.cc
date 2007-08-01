// -*- C++ -*-
//
// Package:    L1ExtraParticleMapProd
// Class:      L1ExtraParticleMapProd
// 
/**\class L1ExtraParticleMapProd \file L1ExtraParticleMapProd.cc L1Trigger/L1ExtraParticleMapProd/src/L1ExtraParticleMapProd.cc \author Werner Sun
*/
//
// Original Author:  Werner Sun
//         Created:  Mon Oct 16 23:19:38 EDT 2006
// $Id: L1ExtraParticleMapProd.cc,v 1.6 2007/03/22 13:42:34 wsun Exp $
//
//


// system include files
#include <memory>

// user include files
#include "L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticleMapProd.h"

//#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

static const int kDefault = -1 ;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
L1ExtraParticleMapProd::L1ExtraParticleMapProd(
   const edm::ParameterSet& iConfig)
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
     singleIsoEmMinEt_( iConfig.getParameter< double >( "singleIsoEmMinEt" ) ),
     doubleIsoEmMinEt_( iConfig.getParameter< double >( "doubleIsoEmMinEt" ) ),
     singleRelaxedEmMinEt_( iConfig.getParameter< double >(
	"singleRelaxedEmMinEt" ) ),
     doubleRelaxedEmMinEt_( iConfig.getParameter< double >(
	"doubleRelaxedEmMinEt" ) ),
     singleMuonMinEt_( iConfig.getParameter< double >( "singleMuonMinEt" ) ),
     doubleMuonMinEt_( iConfig.getParameter< double >( "doubleMuonMinEt" ) ),
     singleTauMinEt_( iConfig.getParameter< double >( "singleTauMinEt" ) ),
     doubleTauMinEt_( iConfig.getParameter< double >( "doubleTauMinEt" ) ),
     singleJetMinEt_( iConfig.getParameter< double >( "singleJetMinEt" ) ),
     doubleJetMinEt_( iConfig.getParameter< double >( "doubleJetMinEt" ) ),
     tripleJetMinEt_( iConfig.getParameter< double >( "tripleJetMinEt" ) ),
     quadJetMinEt_( iConfig.getParameter< double >( "quadJetMinEt" ) ),
     htMin_( iConfig.getParameter< double >( "htMin" ) ),
     metMin_( iConfig.getParameter< double >( "metMin" ) ),
     htMetMinHt_( iConfig.getParameter< double >( "htMetMinHt" ) ),
     htMetMinMet_( iConfig.getParameter< double >( "htMetMinMet" ) ),
     jetMetMinJetEt_( iConfig.getParameter< double >( "jetMetMinJetEt" ) ),
     jetMetMinMet_( iConfig.getParameter< double >( "jetMetMinMet" ) ),
     tauMetMinTauEt_( iConfig.getParameter< double >( "tauMetMinTauEt" ) ),
     tauMetMinMet_( iConfig.getParameter< double >( "tauMetMinMet" ) ),
     muonMetMinMuonEt_( iConfig.getParameter< double >( "muonMetMinMuonEt" ) ),
     muonMetMinMet_( iConfig.getParameter< double >( "muonMetMinMet" ) ),
     isoEmMetMinEmEt_( iConfig.getParameter< double >( "isoEmMetMinEmEt" ) ),
     isoEmMetMinMet_( iConfig.getParameter< double >( "isoEmMetMinMet" ) ),
     muonJetMinMuonEt_( iConfig.getParameter< double >( "muonJetMinMuonEt" ) ),
     muonJetMinJetEt_( iConfig.getParameter< double >( "muonJetMinJetEt" ) ),
     isoEmJetMinEmEt_( iConfig.getParameter< double >( "isoEmJetMinEmEt" ) ),
     isoEmJetMinJetEt_( iConfig.getParameter< double >( "isoEmJetMinJetEt" ) ),
     muonTauMinMuonEt_( iConfig.getParameter< double >( "muonTauMinMuonEt" ) ),
     muonTauMinTauEt_( iConfig.getParameter< double >( "muonTauMinTauEt" ) ),
     muonTauMinDeltaPhi_( iConfig.getParameter< double >(
	"muonTauMinDeltaPhi" ) ),
     muonTauMinDeltaEta_( iConfig.getParameter< double >(
        "muonTauMinDeltaEta" ) ),
     isoEmTauMinEmEt_( iConfig.getParameter< double >( "isoEmTauMinEmEt" ) ),
     isoEmTauMinTauEt_( iConfig.getParameter< double >( "isoEmTauMinTauEt" ) ),
     isoEmTauMinDeltaPhi_( iConfig.getParameter< double >(
	 "isoEmTauMinDeltaPhi" ) ),
     isoEmTauMinDeltaEta_( iConfig.getParameter< double >(
	 "isoEmTauMinDeltaEta" ) ),
     isoEmMuonMinEmEt_( iConfig.getParameter< double >( "isoEmMuonMinEmEt" ) ),
     isoEmMuonMinMuonEt_( iConfig.getParameter< double >(
	"isoEmMuonMinMuonEt" ) ),
     singleJet140Prescale_( iConfig.getParameter< int >(
	"singleJet140Prescale" ) ),
     singleJet60Prescale_( iConfig.getParameter< int >(
	"singleJet60Prescale" ) ),
     singleJet20Prescale_( iConfig.getParameter< int >(
	"singleJet20Prescale" ) ),
     minBiasPrescale_( iConfig.getParameter< int >( "minBiasPrescale" ) )
{
   using namespace l1extra ;

   //register your products
   produces< L1ParticleMapCollection >() ;
   produces< L1GlobalTriggerReadoutRecord >(); 

   //now do what ever other initialization is needed

}


L1ExtraParticleMapProd::~L1ExtraParticleMapProd()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1ExtraParticleMapProd::produce(edm::Event& iEvent,
				const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;
   using namespace l1extra ;


   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // ~~~~~~~~ Get L1Extra particles ~~~~~~~~
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Handle< L1EmParticleCollection > isoEmHandle ;
   iEvent.getByLabel( isoEmSource_, isoEmHandle ) ;

   Handle< L1EmParticleCollection > nonIsoEmHandle ;
   iEvent.getByLabel( nonIsoEmSource_, nonIsoEmHandle ) ;

   Handle< L1JetParticleCollection > cenJetHandle ;
   iEvent.getByLabel( cenJetSource_, cenJetHandle ) ;

   Handle< L1JetParticleCollection > forJetHandle ;
   iEvent.getByLabel( forJetSource_, forJetHandle ) ;

   Handle< L1JetParticleCollection > tauJetHandle ;
   iEvent.getByLabel( tauJetSource_, tauJetHandle ) ;

   Handle< L1MuonParticleCollection > muHandle ;
   iEvent.getByLabel( muonSource_, muHandle ) ;

   Handle< L1EtMissParticle > metHandle ;
   iEvent.getByLabel( etMissSource_, metHandle ) ;

   double met = metHandle->etMiss() ;
   double ht = metHandle->etHad() ;

   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   // ~~~ Evaluate trigger conditions and make a L1ParticleMapCollection. ~~~
   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   // First, form the input vector<Ref>s that will be needed.
   L1EmParticleVectorRef inputIsoEmRefs ;
   addToVectorRefs( isoEmHandle, inputIsoEmRefs ) ;

   L1EmParticleVectorRef inputRelaxedEmRefs ;
   addToVectorRefs( isoEmHandle, inputRelaxedEmRefs ) ;
   addToVectorRefs( nonIsoEmHandle, inputRelaxedEmRefs ) ;

   L1JetParticleVectorRef inputTauRefs ;
   addToVectorRefs( tauJetHandle, inputTauRefs ) ;

   L1JetParticleVectorRef inputJetRefs ;
   addToVectorRefs( forJetHandle, inputJetRefs ) ;
   addToVectorRefs( cenJetHandle, inputJetRefs ) ;
   addToVectorRefs( tauJetHandle, inputJetRefs ) ;

   L1MuonParticleVectorRef inputMuonRefs ;
   addToVectorRefs( muHandle, inputMuonRefs ) ;

   auto_ptr< L1ParticleMapCollection > mapColl( new L1ParticleMapCollection ) ;
   bool globalDecision = false ;

   for( int itrig = 0 ; itrig < L1ParticleMap::kNumOfL1TriggerTypes; ++itrig )
   {
      bool decision = false ;
      std::vector< L1ParticleMap::L1ObjectType > objectTypes ;
      L1EmParticleVectorRef outputEmRefs ;
      L1JetParticleVectorRef outputJetRefs ;
      L1MuonParticleVectorRef outputMuonRefs ;
      L1EtMissParticleRefProd metRef ;
      L1ParticleMap::L1IndexComboVector combos ; // unfilled for single objs

      if( itrig == L1ParticleMap::kSingleIsoEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateSingleObjectTrigger( inputIsoEmRefs,
				      singleIsoEmMinEt_,
				      decision,
				      outputEmRefs ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleSameObjectTrigger( inputIsoEmRefs,
					  doubleIsoEmMinEt_,
					  decision,
					  outputEmRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kSingleRelaxedEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateSingleObjectTrigger( inputRelaxedEmRefs,
				      singleRelaxedEmMinEt_,
				      decision,
				      outputEmRefs ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleRelaxedEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleSameObjectTrigger( inputRelaxedEmRefs,
					  doubleRelaxedEmMinEt_,
					  decision,
					  outputEmRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kSingleMuon )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateSingleObjectTrigger( inputMuonRefs,
				      singleMuonMinEt_,
				      decision,
				      outputMuonRefs ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleMuon )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoubleSameObjectTrigger( inputMuonRefs,
					  doubleMuonMinEt_,
					  decision,
					  outputMuonRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kSingleTau )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateSingleObjectTrigger( inputTauRefs,
				      singleTauMinEt_,
				      decision,
				      outputJetRefs ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleTau )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleSameObjectTrigger( inputTauRefs,
					  doubleTauMinEt_,
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kSingleJet )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateSingleObjectTrigger( inputJetRefs,
				      singleJetMinEt_,
				      decision,
				      outputJetRefs ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleJet )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleSameObjectTrigger( inputJetRefs,
					  doubleJetMinEt_,
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kTripleJet )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateTripleSameObjectTrigger( inputJetRefs,
					  tripleJetMinEt_,
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kQuadJet )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateQuadSameObjectTrigger( inputJetRefs,
					quadJetMinEt_,
					decision,
					outputJetRefs,
					combos ) ;
      }
      else if( itrig == L1ParticleMap::kHT )
      {
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > htMin_ )
	 {
	    decision = true ;
	    metRef = L1EtMissParticleRefProd( metHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kMET )
      {
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > metMin_ )
	 {
	    decision = true ;
	    metRef = L1EtMissParticleRefProd( metHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kHTMET )
      {
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( ht > htMetMinHt_ && met > htMetMinMet_ )
	 {
	    decision = true ;
	    metRef = L1EtMissParticleRefProd( metHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kJetMET )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > jetMetMinMet_ )
	 {
	    evaluateSingleObjectTrigger( inputJetRefs,
					 jetMetMinJetEt_,
					 decision,
					 outputJetRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kTauMET )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > tauMetMinMet_ )
	 {
	    evaluateSingleObjectTrigger( inputTauRefs,
					 tauMetMinTauEt_,
					 decision,
					 outputJetRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kMuonMET )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > muonMetMinMet_ )
	 {
	    evaluateSingleObjectTrigger( inputMuonRefs,
					 muonMetMinMuonEt_,
					 decision,
					 outputMuonRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kIsoEMMET )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > isoEmMetMinMet_ )
	 {
	    evaluateSingleObjectTrigger( inputIsoEmRefs,
					 isoEmMetMinEmEt_,
					 decision,
					 outputEmRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kMuonJet )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefs,
	    inputJetRefs,
	    muonJetMinMuonEt_,
	    muonJetMinJetEt_,
	    0.,
	    0.,
	    decision,
	    outputMuonRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kIsoEMJet )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputIsoEmRefs,
	    inputJetRefs,
	    isoEmJetMinEmEt_,
	    isoEmJetMinJetEt_,
	    0.,
	    0.,
	    decision,
	    outputEmRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kMuonTau )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefs,
	    inputTauRefs,
	    muonTauMinMuonEt_,
	    muonTauMinTauEt_,
	    muonTauMinDeltaPhi_,
	    muonTauMinDeltaEta_,
	    decision,
	    outputMuonRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kIsoEMTau )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputIsoEmRefs,
	    inputTauRefs,
	    isoEmTauMinEmEt_,
	    isoEmTauMinTauEt_,
	    isoEmTauMinDeltaPhi_,
	    isoEmTauMinDeltaEta_,
	    decision,
	    outputEmRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kIsoEMMuon )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputIsoEmRefs,
	    inputMuonRefs,
	    isoEmMuonMinEmEt_,
	    isoEmMuonMinMuonEt_,
	    0.,
	    0.,
	    decision,
	    outputEmRefs,
	    outputMuonRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kSingleJet140 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 L1JetParticleVectorRef outputJetRefsTmp ;
	 evaluateSingleObjectTrigger( inputJetRefs,
				      140.,
				      decision,
				      outputJetRefsTmp ) ;

	 static int singleJet140Counter = 0 ;
	 if( decision )
	 {
	    if( singleJet140Counter % singleJet140Prescale_ )
	    {
	       decision = false ;
	    }
	    else
	    {
	       outputJetRefs = outputJetRefsTmp ;
	    }

	    ++singleJet140Counter ;
	 }
      }
      else if( itrig == L1ParticleMap::kSingleJet60 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 L1JetParticleVectorRef outputJetRefsTmp ;
	 evaluateSingleObjectTrigger( inputJetRefs,
				      60.,
				      decision,
				      outputJetRefsTmp ) ;

	 static int singleJet60Counter = 0 ;
	 if( decision )
	 {
	    if( singleJet60Counter % singleJet60Prescale_ )
	    {
	       decision = false ;
	    }
	    else
	    {
	       outputJetRefs = outputJetRefsTmp ;
	    }

	    ++singleJet60Counter ;
	 }
      }
      else if( itrig == L1ParticleMap::kSingleJet20 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 L1JetParticleVectorRef outputJetRefsTmp ;
	 evaluateSingleObjectTrigger( inputJetRefs,
				      20.,
				      decision,
				      outputJetRefsTmp ) ;

	 static int singleJet20Counter = 0 ;
	 if( decision )
	 {
	    if( singleJet20Counter % singleJet20Prescale_ )
	    {
	       decision = false ;
	    }
	    else
	    {
	       outputJetRefs = outputJetRefsTmp ;
	    }

	    ++singleJet20Counter ;
	 }
      }
      else if( itrig == L1ParticleMap::kMinBias )
      {
	 static int minBiasCounter = 0 ;

	 if( minBiasCounter % minBiasPrescale_ == 0 )
	 {
	    decision = true ;
	 }

	 ++minBiasCounter ;
      }

      // Construct a L1ParticleMap and add it to the collection.
      mapColl->push_back( L1ParticleMap(
	 ( L1ParticleMap::L1TriggerType ) itrig,
	 decision,
	 objectTypes,
	 outputEmRefs,
	 outputJetRefs,
	 outputMuonRefs,
	 metRef,
	 combos ) ) ;

      globalDecision = globalDecision || decision ;
   }

   // Put the L1ParticleMapCollection into the event.
   iEvent.put( mapColl ) ;

   // Make a L1GlobalTriggerReadoutRecord and put it into the event.
   auto_ptr< L1GlobalTriggerReadoutRecord > gtRecord(
      new L1GlobalTriggerReadoutRecord() ) ;
   gtRecord->setDecision( globalDecision ) ;
   iEvent.put( gtRecord ) ;

   return ;
}


template< class TCollection >
void
L1ExtraParticleMapProd::addToVectorRefs(
   const edm::Handle< TCollection >& handle,                // input
   std::vector< edm::Ref< TCollection > >& vectorRefs )     // output
{
   for( size_t i = 0 ; i < handle->size() ; ++i )
   {
      vectorRefs.push_back( edm::Ref< TCollection >( handle, i ) ) ;
   }
}

template< class TCollection >
void
L1ExtraParticleMapProd::evaluateSingleObjectTrigger(
   const std::vector< edm::Ref< TCollection > >& inputRefs, // input
   const double& etThreshold,                               // input
   bool& decision,                                          // output
   std::vector< edm::Ref< TCollection > >& outputRefs )     // output
{
   for( size_t i = 0 ; i < inputRefs.size() ; ++i )
   {
      if( inputRefs[ i ].get()->et() >= etThreshold )
      {
	 decision = true ;
	 outputRefs.push_back( inputRefs[ i ] ) ;
      }
   }
}

template< class TCollection >
void
L1ExtraParticleMapProd::evaluateDoubleSameObjectTrigger(
   const std::vector< edm::Ref< TCollection > >& inputRefs, // input
   const double& etThreshold,                               // input
   bool& decision,                                          // output
   std::vector< edm::Ref< TCollection > >& outputRefs,      // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )     // output
{
   // Use i+1 < inputRefs.size() instead of i < inputRefs.size()-1
   // because i is unsigned, and if size() is 0, then RHS undefined.
   for( size_t i = 0 ; i+1 < inputRefs.size() ; ++i )
   {
      const edm::Ref< TCollection >& refi = inputRefs[ i ] ;
      if( refi.get()->et() >= etThreshold )
      {
	 for( size_t j = i+1 ; j < inputRefs.size() ; ++j )
	 {
	    const edm::Ref< TCollection >& refj = inputRefs[ j ] ;
	    if( refj.get()->et() >= etThreshold )
	    {
	       decision = true ;

	       // If the two objects are already in the list, find
	       // their indices.
	       int iInList = kDefault ;
	       int jInList = kDefault ;
	       for( size_t iout = 0 ; iout < outputRefs.size() ; ++iout )
	       {
		  if( refi == outputRefs[ iout ] )
		  {
		     iInList = iout ;
		  }

		  if( refj == outputRefs[ iout ] )
		  {
		     jInList = iout ;
		  }
	       }

	       // If either object is not in the list, add it, and
	       // record its index.
	       if( iInList == kDefault )
	       {
		  iInList = outputRefs.size() ;
		  outputRefs.push_back( refi ) ;
	       }
		     
	       if( jInList == kDefault )
	       {
		  jInList = outputRefs.size() ;
		  outputRefs.push_back( refj ) ;
	       }

	       // Record this object combination.
	       l1extra::L1ParticleMap::L1IndexCombo combo ;
	       combo.push_back( iInList ) ;
	       combo.push_back( jInList ) ;
	       combos.push_back( combo ) ;
	    }
	 }
      }
   }
}


template< class TCollection >
void
L1ExtraParticleMapProd::evaluateTripleSameObjectTrigger(
   const std::vector< edm::Ref< TCollection > >& inputRefs, // input
   const double& etThreshold,                               // input
   bool& decision,                                          // output
   std::vector< edm::Ref< TCollection > >& outputRefs,      // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )     // output
{
   // Use i+2 < inputRefs.size() instead of i < inputRefs.size()-2
   // because i is unsigned, and if size() is 0, then RHS undefined.
   for( size_t i = 0 ; i+2 < inputRefs.size() ; ++i )
   {
      const edm::Ref< TCollection >& refi = inputRefs[ i ] ;
      if( refi.get()->et() >= etThreshold )
      {
	 for( size_t j = i+1 ; j+1 < inputRefs.size() ; ++j )
	 {
	    const edm::Ref< TCollection >& refj = inputRefs[ j ] ;
	    if( refj.get()->et() >= etThreshold )
	    {
	       for( size_t k = j+1 ; k < inputRefs.size() ; ++k )
	       {
		  const edm::Ref< TCollection >& refk = inputRefs[ k ] ;
		  if( refk.get()->et() >= etThreshold )
		  {
		     decision = true ;

		     // If the three objects are already in the list, find
		     // their indices.
		     int iInList = kDefault ;
		     int jInList = kDefault ;
		     int kInList = kDefault ;
		     for( size_t iout = 0 ; iout < outputRefs.size() ; ++iout )
		     {
			if( refi == outputRefs[ iout ] )
			{
			   iInList = iout ;
			}

			if( refj == outputRefs[ iout ] )
			{
			   jInList = iout ;
			}

			if( refk == outputRefs[ iout ] )
			{
			   kInList = iout ;
			}
		     }

		     // If any object is not in the list, add it, and
		     // record its index.
		     if( iInList == kDefault )
		     {
			iInList = outputRefs.size() ;
			outputRefs.push_back( refi );
		     }
		     
		     if( jInList == kDefault )
		     {
			jInList = outputRefs.size() ;
			outputRefs.push_back( refj );
		     }

		     if( kInList == kDefault )
		     {
			kInList = outputRefs.size() ;
			outputRefs.push_back( refk );
		     }

		     // Record this object combination.
		     l1extra::L1ParticleMap::L1IndexCombo combo ;
		     combo.push_back( iInList ) ;
		     combo.push_back( jInList ) ;
		     combo.push_back( kInList ) ;
		     combos.push_back( combo ) ;
		  }
	       }
	    }
	 }
      }
   }
}


template< class TCollection >
void
L1ExtraParticleMapProd::evaluateQuadSameObjectTrigger(
   const std::vector< edm::Ref< TCollection > >& inputRefs, // input
   const double& etThreshold,                               // input
   bool& decision,                                          // output
   std::vector< edm::Ref< TCollection > >& outputRefs,      // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )     // output
{
   // Use i+3 < inputRefs.size() instead of i < inputRefs.size()-3
   // because i is unsigned, and if size() is 0, then RHS undefined.
   for( size_t i = 0 ; i+3 < inputRefs.size() ; ++i )
   {
      const edm::Ref< TCollection >& refi = inputRefs[ i ] ;
      if( refi.get()->et() >= etThreshold )
      {
	 for( size_t j = i+1 ; j+2 < inputRefs.size() ; ++j )
	 {
	    const edm::Ref< TCollection >& refj = inputRefs[ j ] ;
	    if( refj.get()->et() >= etThreshold )
	    {
	       for( size_t k = j+1 ; k+1 < inputRefs.size() ; ++k )
	       {
		  const edm::Ref< TCollection >& refk = inputRefs[ k ] ;
		  if( refk.get()->et() >= etThreshold )
		  {
		     for( size_t p = k+1 ; p < inputRefs.size() ; ++p )
		     {
			const edm::Ref< TCollection >& refp = inputRefs[ p ] ;
			if( refp.get()->et() >= etThreshold )
			{
			   decision = true ;

			   // If the objects are already in the list, find
			   // their indices.
			   int iInList = kDefault ;
			   int jInList = kDefault ;
			   int kInList = kDefault ;
			   int pInList = kDefault ;
			   for( size_t iout = 0 ;
				iout < outputRefs.size() ; ++iout )
			   {
			      if( refi == outputRefs[ iout ] )
			      {
				 iInList = iout ;
			      }

			      if( refj == outputRefs[ iout ] )
			      {
				 jInList = iout ;
			      }

			      if( refk == outputRefs[ iout ] )
			      {
				 kInList = iout ;
			      }

			      if( refp == outputRefs[ iout ] )
			      {
				 pInList = iout ;
			      }
			   }

			   // If any object is not in the list, add it, and
			   // record its index.
			   if( iInList == kDefault )
			   {
			      iInList = outputRefs.size() ;
			      outputRefs.push_back( refi ) ;
			   }
		     
			   if( jInList == kDefault )
			   {
			      jInList = outputRefs.size() ;
			      outputRefs.push_back( refj ) ;
			   }

			   if( kInList == kDefault )
			   {
			      kInList = outputRefs.size() ;
			      outputRefs.push_back( refk ) ;
			   }

			   if( pInList == kDefault )
			   {
			      pInList = outputRefs.size() ;
			      outputRefs.push_back( refp ) ;
			   }

			   // Record this object combination.
			   l1extra::L1ParticleMap::L1IndexCombo combo ;
			   combo.push_back( iInList ) ;
			   combo.push_back( jInList ) ;
			   combo.push_back( kInList ) ;
			   combo.push_back( pInList ) ;
			   combos.push_back( combo ) ;
			}
		     }
		  }
	       }
	    }
	 }
      }
   }
}


template< class TCollection1, class TCollection2 >
void
L1ExtraParticleMapProd::evaluateDoubleDifferentObjectTrigger(
   const std::vector< edm::Ref< TCollection1 > >& inputRefs1, // input
   const std::vector< edm::Ref< TCollection2 > >& inputRefs2, // input
   const double& etThreshold1,                                // input
   const double& etThreshold2,                                // input
   const double& deltaPhiMin,                                 // input
   const double& deltaEtaMin,                                 // input
   bool& decision,                                            // output
   std::vector< edm::Ref< TCollection1 > >& outputRefs1,      // output
   std::vector< edm::Ref< TCollection2 > >& outputRefs2,      // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )       // output
{
   for( size_t i = 0 ; i < inputRefs1.size() ; ++i )
   {
      const edm::Ref< TCollection1 >& refi = inputRefs1[ i ] ;
      if( refi.get()->et() >= etThreshold1 )
      {
 	 double phi1 = refi.get()->phi() ;
	 double eta1 = refi.get()->eta() ;

	 for( size_t j = 0 ; j < inputRefs2.size() ; ++j )
	 {
	    const edm::Ref< TCollection2 >& refj = inputRefs2[ j ] ;

	    double phi2 = refj.get()->phi() ;
	    double eta2 = refj.get()->eta() ;
	    double deltaPhi = fabs( phi1 - phi2 ) ;
	    deltaPhi =
	       ( deltaPhi > M_PI ) ? fabs( deltaPhi - 2. * M_PI ) : deltaPhi ;

	    if( refj.get()->et() >= etThreshold2 &&
		( deltaPhi > deltaPhiMin ||
		  fabs( eta1 - eta2 ) > deltaEtaMin ) )
	    {
	       decision = true ;

	       // If the two objects are already in their respective lists,
	       // find their indices.
	       int iInList = kDefault ;
	       for( size_t iout = 0 ; iout < outputRefs1.size() ; ++iout )
	       {
		  if( refi == outputRefs1[ iout ] )
		  {
		     iInList = iout ;
		  }
	       }

	       int jInList = kDefault ;
	       for( size_t jout = 0 ; jout < outputRefs2.size() ; ++jout )
	       {
		  if( refj == outputRefs2[ jout ] )
		  {
		     jInList = jout ;
		  }
	       }

	       // If either object is not in the list, add it, and
	       // record its index.
	       if( iInList == kDefault )
	       {
		  iInList = outputRefs1.size() ;
		  outputRefs1.push_back( refi ) ;
	       }
		     
	       if( jInList == kDefault )
	       {
		  jInList = outputRefs2.size() ;
		  outputRefs2.push_back( refj ) ;
	       }

	       // Record this object combination.
	       l1extra::L1ParticleMap::L1IndexCombo combo ;
	       combo.push_back( iInList ) ;
	       combo.push_back( jInList ) ;
	       combos.push_back( combo ) ;
	    }
	 }
      }
   }
}


//define this as a plug-in
//DEFINE_FWK_MODULE(L1ExtraParticleMapProd);
