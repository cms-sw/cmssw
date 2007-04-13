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
// $Id: L1ExtraParticleMapProd.cc,v 1.7 2007/04/02 08:03:15 wsun Exp $
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
   : muonSource_( iConfig.getParameter< edm::InputTag >(
      "muonSource" ) ),
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
     etMissSource_( iConfig.getParameter< edm::InputTag >(
	"etMissSource" ) )
{
   using namespace l1extra ;

   //register your products
   produces< L1ParticleMapCollection >() ;
   produces< L1GlobalTriggerReadoutRecord >(); 

   //now do what ever other initialization is needed
   for( int i = 0 ; i < L1ParticleMap::kNumOfL1TriggerTypes ; ++i )
   {
      singleThresholds_[ i ] = 0. ;
      doubleThresholds_[ i ].first = 0. ;
      doubleThresholds_[ i ].second = 0. ;
      prescaleCounters_[ i ] = 0 ;
      prescales_[ i ] = 1 ;
   }

   // Single object triggers, 5 thresholds each

   singleThresholds_[ L1ParticleMap::kSingleMuon ] =
      iConfig.getParameter< double >( "singleMuonThresh" ) ;
   prescales_[ L1ParticleMap::kSingleMuon ] =
      iConfig.getParameter< int >( "singleMuonPrescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMuon_A ] =
      iConfig.getParameter< double >( "singleMuonThresh_A" ) ;
   prescales_[ L1ParticleMap::kSingleMuon_A ] =
      iConfig.getParameter< int >( "singleMuonPrescale_A" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMuon_B ] =
      iConfig.getParameter< double >( "singleMuonThresh_B" ) ;
   prescales_[ L1ParticleMap::kSingleMuon_B ] =
      iConfig.getParameter< int >( "singleMuonPrescale_B" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMuon_C ] =
      iConfig.getParameter< double >( "singleMuonThresh_C" ) ;
   prescales_[ L1ParticleMap::kSingleMuon_C ] =
      iConfig.getParameter< int >( "singleMuonPrescale_C" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMuon_D ] =
      iConfig.getParameter< double >( "singleMuonThresh_D" ) ;
   prescales_[ L1ParticleMap::kSingleMuon_D ] =
      iConfig.getParameter< int >( "singleMuonPrescale_D" ) ;

   singleThresholds_[ L1ParticleMap::kSingleIsoEM ] =
      iConfig.getParameter< double >( "singleIsoEMThresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEM ] =
      iConfig.getParameter< int >( "singleIsoEMPrescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEM_A ] =
      iConfig.getParameter< double >( "singleIsoEMThresh_A" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEM_A ] =
      iConfig.getParameter< int >( "singleIsoEMPrescale_A" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEM_B ] =
      iConfig.getParameter< double >( "singleIsoEMThresh_B" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEM_B ] =
      iConfig.getParameter< int >( "singleIsoEMPrescale_B" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEM_C ] =
      iConfig.getParameter< double >( "singleIsoEMThresh_C" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEM_C ] =
      iConfig.getParameter< int >( "singleIsoEMPrescale_C" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEM_D ] =
      iConfig.getParameter< double >( "singleIsoEMThresh_D" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEM_D ] =
      iConfig.getParameter< int >( "singleIsoEMPrescale_D" ) ;

   singleThresholds_[ L1ParticleMap::kSingleRelaxedEM ] =
      iConfig.getParameter< double >( "singleRelaxedEMThresh" ) ;
   prescales_[ L1ParticleMap::kSingleRelaxedEM ] =
      iConfig.getParameter< int >( "singleRelaxedEMPrescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleRelaxedEM_A ] =
      iConfig.getParameter< double >( "singleRelaxedEMThresh_A" ) ;
   prescales_[ L1ParticleMap::kSingleRelaxedEM_A ] =
      iConfig.getParameter< int >( "singleRelaxedEMPrescale_A" ) ;
   singleThresholds_[ L1ParticleMap::kSingleRelaxedEM_B ] =
      iConfig.getParameter< double >( "singleRelaxedEMThresh_B" ) ;
   prescales_[ L1ParticleMap::kSingleRelaxedEM_B ] =
      iConfig.getParameter< int >( "singleRelaxedEMPrescale_B" ) ;
   singleThresholds_[ L1ParticleMap::kSingleRelaxedEM_C ] =
      iConfig.getParameter< double >( "singleRelaxedEMThresh_C" ) ;
   prescales_[ L1ParticleMap::kSingleRelaxedEM_C ] =
      iConfig.getParameter< int >( "singleRelaxedEMPrescale_C" ) ;
   singleThresholds_[ L1ParticleMap::kSingleRelaxedEM_D ] =
      iConfig.getParameter< double >( "singleRelaxedEMThresh_D" ) ;
   prescales_[ L1ParticleMap::kSingleRelaxedEM_D ] =
      iConfig.getParameter< int >( "singleRelaxedEMPrescale_D" ) ;

   singleThresholds_[ L1ParticleMap::kSingleJet ] =
      iConfig.getParameter< double >( "singleJetThresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet ] =
      iConfig.getParameter< int >( "singleJetPrescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet_A ] =
      iConfig.getParameter< double >( "singleJetThresh_A" ) ;
   prescales_[ L1ParticleMap::kSingleJet_A ] =
      iConfig.getParameter< int >( "singleJetPrescale_A" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet_B ] =
      iConfig.getParameter< double >( "singleJetThresh_B" ) ;
   prescales_[ L1ParticleMap::kSingleJet_B ] =
      iConfig.getParameter< int >( "singleJetPrescale_B" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet_C ] =
      iConfig.getParameter< double >( "singleJetThresh_C" ) ;
   prescales_[ L1ParticleMap::kSingleJet_C ] =
      iConfig.getParameter< int >( "singleJetPrescale_C" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet_D ] =
      iConfig.getParameter< double >( "singleJetThresh_D" ) ;
   prescales_[ L1ParticleMap::kSingleJet_D ] =
      iConfig.getParameter< int >( "singleJetPrescale_D" ) ;

   singleThresholds_[ L1ParticleMap::kSingleTau ] =
      iConfig.getParameter< double >( "singleTauThresh" ) ;
   prescales_[ L1ParticleMap::kSingleTau ] =
      iConfig.getParameter< int >( "singleTauPrescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTau_A ] =
      iConfig.getParameter< double >( "singleTauThresh_A" ) ;
   prescales_[ L1ParticleMap::kSingleTau_A ] =
      iConfig.getParameter< int >( "singleTauPrescale_A" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTau_B ] =
      iConfig.getParameter< double >( "singleTauThresh_B" ) ;
   prescales_[ L1ParticleMap::kSingleTau_B ] =
      iConfig.getParameter< int >( "singleTauPrescale_B" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTau_C ] =
      iConfig.getParameter< double >( "singleTauThresh_C" ) ;
   prescales_[ L1ParticleMap::kSingleTau_C ] =
      iConfig.getParameter< int >( "singleTauPrescale_C" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTau_D ] =
      iConfig.getParameter< double >( "singleTauThresh_D" ) ;
   prescales_[ L1ParticleMap::kSingleTau_D ] =
      iConfig.getParameter< int >( "singleTauPrescale_D" ) ;

   singleThresholds_[ L1ParticleMap::kHT ] =
      iConfig.getParameter< double >( "htMin" ) ;
   prescales_[ L1ParticleMap::kHT ] =
      iConfig.getParameter< int >( "htPrescale" ) ;
   singleThresholds_[ L1ParticleMap::kHT_A ] =
      iConfig.getParameter< double >( "htMin_A" ) ;
   prescales_[ L1ParticleMap::kHT_A ] =
      iConfig.getParameter< int >( "htPrescale_A" ) ;
   singleThresholds_[ L1ParticleMap::kHT_B ] =
      iConfig.getParameter< double >( "htMin_B" ) ;
   prescales_[ L1ParticleMap::kHT_B ] =
      iConfig.getParameter< int >( "htPrescale_B" ) ;
   singleThresholds_[ L1ParticleMap::kHT_C ] =
      iConfig.getParameter< double >( "htMin_C" ) ;
   prescales_[ L1ParticleMap::kHT_C ] =
      iConfig.getParameter< int >( "htPrescale_C" ) ;
   singleThresholds_[ L1ParticleMap::kHT_D ] =
      iConfig.getParameter< double >( "htMin_D" ) ;
   prescales_[ L1ParticleMap::kHT_D ] =
      iConfig.getParameter< int >( "htPrescale_D" ) ;

   singleThresholds_[ L1ParticleMap::kMET ] =
      iConfig.getParameter< double >( "metMin" ) ;
   prescales_[ L1ParticleMap::kMET ] =
      iConfig.getParameter< int >( "metPrescale" ) ;
   singleThresholds_[ L1ParticleMap::kMET_A ] =
      iConfig.getParameter< double >( "metMin_A" ) ;
   prescales_[ L1ParticleMap::kMET_A ] =
      iConfig.getParameter< int >( "metPrescale_A" ) ;
   singleThresholds_[ L1ParticleMap::kMET_B ] =
      iConfig.getParameter< double >( "metMin_B" ) ;
   prescales_[ L1ParticleMap::kMET_B ] =
      iConfig.getParameter< int >( "metPrescale_B" ) ;
   singleThresholds_[ L1ParticleMap::kMET_C ] =
      iConfig.getParameter< double >( "metMin_C" ) ;
   prescales_[ L1ParticleMap::kMET_C ] =
      iConfig.getParameter< int >( "metPrescale_C" ) ;
   singleThresholds_[ L1ParticleMap::kMET_D ] =
      iConfig.getParameter< double >( "metMin_D" ) ;
   prescales_[ L1ParticleMap::kMET_D ] =
      iConfig.getParameter< int >( "metPrescale_D" ) ;

   // AA triggers

   singleThresholds_[ L1ParticleMap::kDoubleMuon ] =
      iConfig.getParameter< double >( "doubleMuonThresh" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleIsoEM ] =
      iConfig.getParameter< double >( "doubleIsoEMThresh" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleRelaxedEM ] =
      iConfig.getParameter< double >( "doubleRelaxedEMThresh" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleJet ] =
      iConfig.getParameter< double >( "doubleJetThresh" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleTau ] =
      iConfig.getParameter< double >( "doubleTauThresh" ) ;

   // AB triggers

   doubleThresholds_[ L1ParticleMap::kMuonIsoEM ].first =
      iConfig.getParameter< double >( "muonIsoEMThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonIsoEM ].second =
      iConfig.getParameter< double >( "muonIsoEMThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonRelaxedEM ].first =
      iConfig.getParameter< double >( "muonRelaxedEMThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonRelaxedEM ].second =
      iConfig.getParameter< double >( "muonRelaxedEMThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonJet ].first =
      iConfig.getParameter< double >( "muonJetThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonJet ].second =
      iConfig.getParameter< double >( "muonJetThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonTau ].first =
      iConfig.getParameter< double >( "muonTauThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonTau ].second =
      iConfig.getParameter< double >( "muonTauThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMRelaxedEM ].first =
      iConfig.getParameter< double >( "isoEMRelaxedEMThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMRelaxedEM ].second =
      iConfig.getParameter< double >( "isoEMRelaxedEMThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMJet ].first =
      iConfig.getParameter< double >( "isoEMJetThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMJet ].second =
      iConfig.getParameter< double >( "isoEMJetThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMTau ].first =
      iConfig.getParameter< double >( "isoEMTauThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMTau ].second =
      iConfig.getParameter< double >( "isoEMTauThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kRelaxedEMJet ].first =
      iConfig.getParameter< double >( "relaxedEMJetThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kRelaxedEMJet ].second =
      iConfig.getParameter< double >( "relaxedEMJetThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kRelaxedEMTau ].first =
      iConfig.getParameter< double >( "relaxedEMTauThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kRelaxedEMTau ].second =
      iConfig.getParameter< double >( "relaxedEMTauThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kJetTau ].first =
      iConfig.getParameter< double >( "jetTauThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJetTau ].second =
      iConfig.getParameter< double >( "jetTauThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonHT ].first =
      iConfig.getParameter< double >( "muonHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonHT ].second =
      iConfig.getParameter< double >( "muonHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMHT ].first =
      iConfig.getParameter< double >( "isoEMHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMHT ].second =
      iConfig.getParameter< double >( "isoEMHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kRelaxedEMHT ].first =
      iConfig.getParameter< double >( "relaxedEMHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kRelaxedEMHT ].second =
      iConfig.getParameter< double >( "relaxedEMHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kJetHT ].first =
      iConfig.getParameter< double >( "jetHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJetHT ].second =
      iConfig.getParameter< double >( "jetHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kTauHT ].first =
      iConfig.getParameter< double >( "tauHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kTauHT ].second =
      iConfig.getParameter< double >( "tauHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonMET ].first =
      iConfig.getParameter< double >( "muonMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMuonMET ].second =
      iConfig.getParameter< double >( "muonMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMMET ].first =
      iConfig.getParameter< double >( "isoEMMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEMMET ].second =
      iConfig.getParameter< double >( "isoEMMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kRelaxedEMMET ].first =
      iConfig.getParameter< double >( "relaxedEMMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kRelaxedEMMET ].second =
      iConfig.getParameter< double >( "relaxedEMMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kJetMET ].first =
      iConfig.getParameter< double >( "jetMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJetMET ].second =
      iConfig.getParameter< double >( "jetMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kTauMET ].first =
      iConfig.getParameter< double >( "tauMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kTauMET ].second =
      iConfig.getParameter< double >( "tauMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kHTMET ].first =
      iConfig.getParameter< double >( "htMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kHTMET ].second =
      iConfig.getParameter< double >( "htMETThresh2" ) ;

   // AAA triggers

   singleThresholds_[ L1ParticleMap::kTripleMuon ] =
      iConfig.getParameter< double >( "tripleMuonThresh" ) ;
   singleThresholds_[ L1ParticleMap::kTripleIsoEM ] =
      iConfig.getParameter< double >( "tripleIsoEMThresh" ) ;
   singleThresholds_[ L1ParticleMap::kTripleRelaxedEM ] =
      iConfig.getParameter< double >( "tripleRelaxedEMThresh" ) ;
   singleThresholds_[ L1ParticleMap::kTripleJet ] =
      iConfig.getParameter< double >( "tripleJetThresh" ) ;
   singleThresholds_[ L1ParticleMap::kTripleTau ] =
      iConfig.getParameter< double >( "tripleTauThresh" ) ;

   // AAB triggers

   doubleThresholds_[ L1ParticleMap::kDoubleMuonIsoEM ].first =
      iConfig.getParameter< double >( "doubleMuonIsoEMThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMuonIsoEM ].second =
      iConfig.getParameter< double >( "doubleMuonIsoEMThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMuonRelaxedEM ].first =
      iConfig.getParameter< double >( "doubleMuonRelaxedEMThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMuonRelaxedEM ].second =
      iConfig.getParameter< double >( "doubleMuonRelaxedEMThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEMMuon ].first =
      iConfig.getParameter< double >( "doubleIsoEMMuonThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEMMuon ].second =
      iConfig.getParameter< double >( "doubleIsoEMMuonThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleRelaxedEMMuon ].first =
      iConfig.getParameter< double >( "doubleRelaxedEMMuonThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleRelaxedEMMuon ].second =
      iConfig.getParameter< double >( "doubleRelaxedEMMuonThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMuonHT ].first =
      iConfig.getParameter< double >( "doubleMuonHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMuonHT ].second =
      iConfig.getParameter< double >( "doubleMuonHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEMHT ].first =
      iConfig.getParameter< double >( "doubleIsoEMHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEMHT ].second =
      iConfig.getParameter< double >( "doubleIsoEMHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleRelaxedEMHT ].first =
      iConfig.getParameter< double >( "doubleRelaxedEMHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleRelaxedEMHT ].second =
      iConfig.getParameter< double >( "doubleRelaxedEMHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJetHT ].first =
      iConfig.getParameter< double >( "doubleJetHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJetHT ].second =
      iConfig.getParameter< double >( "doubleJetHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauHT ].first =
      iConfig.getParameter< double >( "doubleTauHTThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauHT ].second =
      iConfig.getParameter< double >( "doubleTauHTThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMuonMET ].first =
      iConfig.getParameter< double >( "doubleMuonMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMuonMET ].second =
      iConfig.getParameter< double >( "doubleMuonMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEMMET ].first =
      iConfig.getParameter< double >( "doubleIsoEMMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEMMET ].second =
      iConfig.getParameter< double >( "doubleIsoEMMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleRelaxedEMMET ].first =
      iConfig.getParameter< double >( "doubleRelaxedEMMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleRelaxedEMMET ].second =
      iConfig.getParameter< double >( "doubleRelaxedEMMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJetMET ].first =
      iConfig.getParameter< double >( "doubleJetMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJetMET ].second =
      iConfig.getParameter< double >( "doubleJetMETThresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauMET ].first =
      iConfig.getParameter< double >( "doubleTauMETThresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauMET ].second =
      iConfig.getParameter< double >( "doubleTauMETThresh2" ) ;

   singleThresholds_[ L1ParticleMap::kQuadJet ] =
      iConfig.getParameter< double >( "quadJetThresh" ) ;

   prescales_[ L1ParticleMap::kMinBias ] =
      iConfig.getParameter< int >( "minBiasPrescale" ) ;
   prescales_[ L1ParticleMap::kZeroBias ] =
      iConfig.getParameter< int >( "zeroBiasPrescale" ) ;

   for( int i = 0 ; i < L1ParticleMap::kNumOfL1TriggerTypes ; ++i )
   {
      std::cout
	 << "|  "
	 << i
	 << "  |  "
	 << L1ParticleMap::triggerName( ( L1ParticleMap::L1TriggerType ) i )
	 << "  |  " ;

      if( singleThresholds_[ i ] > 0 )
      {
	 std::cout << singleThresholds_[ i ] ;
      }
      else if( doubleThresholds_[ i ].first > 0 )
      {
	 std::cout << doubleThresholds_[ i ].first << ", "
		   << doubleThresholds_[ i ].second ;
      }
      else
      {
	 std::cout << "---" ;
      }

      std::cout << "  |  "
		<< prescales_[ i ]
		<< "  |"
		<< std::endl ;
   }
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
   std::vector< bool > decisionWord ;

   for( int itrig = 0 ; itrig < L1ParticleMap::kNumOfL1TriggerTypes; ++itrig )
   {
      bool decision = false ;
      std::vector< L1ParticleMap::L1ObjectType > objectTypes ;
      L1EmParticleVectorRef outputEmRefs ;
      L1JetParticleVectorRef outputJetRefs ;
      L1MuonParticleVectorRef outputMuonRefs ;
      L1EtMissParticleRefProd metRef ;
      L1ParticleMap::L1IndexComboVector combos ; // unfilled for single objs

      if( itrig >= L1ParticleMap::kSingleMuon &&
	  itrig <= L1ParticleMap::kSingleMuon_D )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateSingleObjectTrigger( inputMuonRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputMuonRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kSingleIsoEM &&
	       itrig <= L1ParticleMap::kSingleIsoEM_D )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateSingleObjectTrigger( inputIsoEmRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputEmRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kSingleRelaxedEM &&
	       itrig <= L1ParticleMap::kSingleRelaxedEM_D )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateSingleObjectTrigger( inputRelaxedEmRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputEmRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kSingleJet &&
	       itrig <= L1ParticleMap::kSingleJet_D )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateSingleObjectTrigger( inputJetRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputJetRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kSingleTau &&
	       itrig <= L1ParticleMap::kSingleTau_D )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateSingleObjectTrigger( inputTauRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputJetRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kHT &&
	       itrig <= L1ParticleMap::kHT_D )
      {
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 L1EtMissParticleRefProd metRefTmp ;
	 if( ht > singleThresholds_[ itrig ] )
	 {
	    decision = true ;
	    metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	 }

	 if( decision )
	 {
	    if( prescaleCounters_[ itrig ] % prescales_[ itrig ] )
	    {
	       decision = false ;
	    }
	    else
	    {
	       metRef = metRefTmp ;
	    }

	    ++prescaleCounters_[ itrig ] ;
	 }
      }
      else if( itrig >= L1ParticleMap::kMET &&
	       itrig <= L1ParticleMap::kMET_D )
      {
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 L1EtMissParticleRefProd metRefTmp ;
	 if( met > singleThresholds_[ itrig ] )
	 {
	    decision = true ;
	    metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	 }

	 if( decision )
	 {
	    if( prescaleCounters_[ itrig ] % prescales_[ itrig ] )
	    {
	       decision = false ;
	    }
	    else
	    {
	       metRef = metRefTmp ;
	    }

	    ++prescaleCounters_[ itrig ] ;
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleMuon )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoubleSameObjectTrigger( inputMuonRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputMuonRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleSameObjectTrigger( inputIsoEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleRelaxedEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleSameObjectTrigger( inputRelaxedEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleJet )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleSameObjectTrigger( inputJetRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleTau )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleSameObjectTrigger( inputTauRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kMuonIsoEM )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefs,
	    inputIsoEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefs,
	    outputEmRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kMuonRelaxedEM )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefs,
	    inputRelaxedEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefs,
	    outputEmRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kMuonJet )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefs,
	    inputJetRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefs,
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
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kIsoEMRelaxedEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleDifferentObjectSameTypeTrigger(
	    inputIsoEmRefs,
	    inputRelaxedEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kIsoEMJet )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentCaloObjectTrigger(
	    inputIsoEmRefs,
	    inputJetRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kIsoEMTau )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentCaloObjectTrigger(
	    inputIsoEmRefs,
	    inputTauRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kRelaxedEMJet )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentCaloObjectTrigger(
	    inputRelaxedEmRefs,
	    inputJetRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kRelaxedEMTau )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentCaloObjectTrigger(
	    inputRelaxedEmRefs,
	    inputTauRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefs,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kJetTau )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectSameTypeTrigger(
	    inputJetRefs,
	    inputTauRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputJetRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kMuonHT )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputMuonRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
					 decision,
					 outputMuonRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kIsoEMHT )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputIsoEmRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
					 decision,
					 outputEmRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kRelaxedEMHT )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputRelaxedEmRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
					 decision,
					 outputEmRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kJetHT )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputJetRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
					 decision,
					 outputJetRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kTauHT )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputTauRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
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

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputMuonRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
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

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputIsoEmRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
					 decision,
					 outputEmRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kRelaxedEMMET )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputRelaxedEmRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
					 decision,
					 outputEmRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kJetMET )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputJetRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
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

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputTauRefs,
					 doubleThresholds_[ itrig ].first,
					 prescaleCounters_[ itrig ],
					 1, // prescale
					 decision,
					 outputJetRefs ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kHTMET )
      {
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( ht > doubleThresholds_[ itrig ].first &&
	     met > doubleThresholds_[ itrig ].second )
	 {
	    decision = true ;
	    metRef = L1EtMissParticleRefProd( metHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kTripleMuon )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateTripleSameObjectTrigger( inputMuonRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputMuonRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kTripleIsoEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateTripleSameObjectTrigger( inputIsoEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kTripleRelaxedEM )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateTripleSameObjectTrigger( inputRelaxedEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kTripleJet )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateTripleSameObjectTrigger( inputJetRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kTripleTau )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateTripleSameObjectTrigger( inputTauRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleMuonIsoEM )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoublePlusSingleObjectTrigger(
	    inputMuonRefs,
	    inputIsoEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefs,
	    outputEmRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleMuonRelaxedEM )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoublePlusSingleObjectTrigger(
	    inputMuonRefs,
	    inputRelaxedEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefs,
	    outputEmRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEMMuon )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoublePlusSingleObjectTrigger(
	    inputIsoEmRefs,
	    inputMuonRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefs,
	    outputMuonRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleRelaxedEMMuon )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoublePlusSingleObjectTrigger(
	    inputRelaxedEmRefs,
	    inputMuonRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefs,
	    outputMuonRefs,
	    combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleMuonHT )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputMuonRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputMuonRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEMHT )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputIsoEmRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputEmRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleRelaxedEMHT )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputRelaxedEmRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputEmRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleJetHT )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputJetRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputJetRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleTauHT )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ht > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputTauRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputJetRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleMuonMET )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputMuonRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputMuonRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEMMET )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputIsoEmRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputEmRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleRelaxedEMMET )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputRelaxedEmRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputEmRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleJetMET )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputJetRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputJetRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleTauMET )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met > doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputTauRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputJetRefs,
					     combos,
					     true ) ;

	    if( decision )
	    {
	       metRef = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kQuadJet )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateQuadSameObjectTrigger( inputJetRefs,
					singleThresholds_[ itrig ],
					decision,
					outputJetRefs,
					combos ) ;
      }
//       else if( itrig == L1ParticleMap::VBF_A )
//       {
	 
//       }


      else if( itrig == L1ParticleMap::kMinBias )
      {
	 if( ht >= 10. && 
	     prescaleCounters_[ itrig ] % prescales_[ itrig ] == 0 )
	 {
	    decision = true ;
	 }

	 ++prescaleCounters_[ itrig ] ;
      }
      else if( itrig == L1ParticleMap::kZeroBias )
      {
	 if( prescaleCounters_[ itrig ] % prescales_[ itrig ] == 0 )
	 {
	    decision = true ;
	 }

	 ++prescaleCounters_[ itrig ] ;
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
      decisionWord.push_back( decision ) ;
   }

   // Put the L1ParticleMapCollection into the event.
   iEvent.put( mapColl ) ;

   // Make a L1GlobalTriggerReadoutRecord and put it into the event.
   auto_ptr< L1GlobalTriggerReadoutRecord > gtRecord(
      new L1GlobalTriggerReadoutRecord() ) ;
   gtRecord->setDecision( globalDecision ) ;
   gtRecord->setDecisionWord( decisionWord ) ;
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
   int& prescaleCounter,                                    // input
   const int& prescale,                                     // input
   bool& decision,                                          // output
   std::vector< edm::Ref< TCollection > >& outputRefs )     // output
{
   std::vector< edm::Ref< TCollection > > outputRefsTmp ;

   for( size_t i = 0 ; i < inputRefs.size() ; ++i )
   {
      if( inputRefs[ i ].get()->et() >= etThreshold )
      {
	 decision = true ;
	 outputRefsTmp.push_back( inputRefs[ i ] ) ;
      }
   }

   if( decision )
   {
      if( prescaleCounter % prescale )
      {
	 decision = false ;
      }
      else
      {
	 outputRefs = outputRefsTmp ;
      }

      ++prescaleCounter ;
   }
}

template< class TCollection >
void
L1ExtraParticleMapProd::evaluateDoubleSameObjectTrigger(
   const std::vector< edm::Ref< TCollection > >& inputRefs, // input
   const double& etThreshold,                               // input
   bool& decision,                                          // output
   std::vector< edm::Ref< TCollection > >& outputRefs,      // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos,      // output
   bool combinedWithGlobalObject )                          // input
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
	       if( combinedWithGlobalObject ) combo.push_back( 0 ) ;
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


template< class TCollection1, class TCollection2 >
void
L1ExtraParticleMapProd::evaluateDoublePlusSingleObjectTrigger(
   const std::vector< edm::Ref< TCollection1 > >& inputRefs1, // input
   const std::vector< edm::Ref< TCollection2 > >& inputRefs2, // input
   const double& etThreshold1,                                // input
   const double& etThreshold2,                                // input
   bool& decision,                                            // output
   std::vector< edm::Ref< TCollection1 > >& outputRefs1,      // output
   std::vector< edm::Ref< TCollection2 > >& outputRefs2,      // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )       // output
{
   // Use i+1 < inputRefs.size() instead of i < inputRefs.size()-1
   // because i is unsigned, and if size() is 0, then RHS undefined.
   for( size_t i = 0 ; i+1 < inputRefs1.size() ; ++i )
   {
      const edm::Ref< TCollection1 >& refi = inputRefs1[ i ] ;
      if( refi.get()->et() >= etThreshold1 )
      {
	 for( size_t j = i+1 ; j < inputRefs1.size() ; ++j )
	 {
	    const edm::Ref< TCollection1 >& refj = inputRefs1[ j ] ;
	    if( refj.get()->et() >= etThreshold1 )
	    {
	       for( size_t k = 0 ; k < inputRefs2.size() ; ++k )
	       {
		  const edm::Ref< TCollection2 >& refk = inputRefs2[ k ] ;
		  if( refk.get()->et() >= etThreshold2 )
		  {
		     decision = true ;

		     // If the three objects are already in the list, find
		     // their indices.
		     int iInList = kDefault ;
		     int jInList = kDefault ;

		     for( size_t iout = 0 ;
			  iout < outputRefs1.size() ; ++iout )
		     {
			if( refi == outputRefs1[ iout ] )
			{
			   iInList = iout ;
			}

			if( refj == outputRefs1[ iout ] )
			{
			   jInList = iout ;
			}
		     }

		     int kInList = kDefault ;
		     for( size_t kout = 0 ;
			  kout < outputRefs2.size() ; ++kout )
		     {
			if( refk == outputRefs2[ kout ] )
			{
			   kInList = kout ;
			}
		     }

		     // If any object is not in the list, add it, and
		     // record its index.
		     if( iInList == kDefault )
		     {
			iInList = outputRefs1.size() ;
			outputRefs1.push_back( refi );
		     }
		     
		     if( jInList == kDefault )
		     {
			jInList = outputRefs1.size() ;
			outputRefs1.push_back( refj );
		     }

		     if( kInList == kDefault )
		     {
			kInList = outputRefs2.size() ;
			outputRefs2.push_back( refk );
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
	 for( size_t j = 0 ; j < inputRefs2.size() ; ++j )
	 {
	    const edm::Ref< TCollection2 >& refj = inputRefs2[ j ] ;

	    if( refj.get()->et() >= etThreshold2 )
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


template< class TCollection >
void
L1ExtraParticleMapProd::evaluateDoubleDifferentObjectSameTypeTrigger(
   const std::vector< edm::Ref< TCollection > >& inputRefs1, // input
   const std::vector< edm::Ref< TCollection > >& inputRefs2, // input
   const double& etThreshold1,                               // input
   const double& etThreshold2,                               // input
   bool& decision,                                           // output
   std::vector< edm::Ref< TCollection > >& outputRefs,       // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )      // output
{
   for( size_t i = 0 ; i < inputRefs1.size() ; ++i )
   {
      const edm::Ref< TCollection >& refi = inputRefs1[ i ] ;
      if( refi.get()->et() >= etThreshold1 )
      {
	 for( size_t j = 0 ; j < inputRefs2.size() ; ++j )
	 {
	    const edm::Ref< TCollection >& refj = inputRefs2[ j ] ;

	    if( refj.get()->et() >= etThreshold2 &&
		refi != refj )
	    {
	       decision = true ;

	       // If the two objects are already in their respective lists,
	       // find their indices.
	       int iInList = kDefault ;
	       for( size_t iout = 0 ; iout < outputRefs.size() ; ++iout )
	       {
		  if( refi == outputRefs[ iout ] )
		  {
		     iInList = iout ;
		  }
	       }

	       int jInList = kDefault ;
	       for( size_t jout = 0 ; jout < outputRefs.size() ; ++jout )
	       {
		  if( refj == outputRefs[ jout ] )
		  {
		     jInList = jout ;
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

void
L1ExtraParticleMapProd::evaluateDoubleDifferentCaloObjectTrigger(
   const l1extra::L1EmParticleVectorRef& inputRefs1,         // input
   const l1extra::L1JetParticleVectorRef& inputRefs2,        // input
   const double& etThreshold1,                                // input
   const double& etThreshold2,                                // input
   bool& decision,                                            // output
   l1extra::L1EmParticleVectorRef& outputRefs1,               // output
   l1extra::L1JetParticleVectorRef& outputRefs2,              // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )       // output
{
   for( size_t i = 0 ; i < inputRefs1.size() ; ++i )
   {
      const l1extra::L1EmParticleRef& refi = inputRefs1[ i ] ;
      if( refi.get()->et() >= etThreshold1 )
      {
	 for( size_t j = 0 ; j < inputRefs2.size() ; ++j )
	 {
	    const l1extra::L1JetParticleRef& refj = inputRefs2[ j ] ;

	    // Check for identical region only if both HW objects are non-null.
	    if( refj.get()->et() >= etThreshold2 &&
		( refi.get()->gctEmCand() == 0 ||
		  refj.get()->gctJetCand() == 0 ||
		  refi.get()->gctEmCand()->regionId() !=
		  refj.get()->gctJetCand()->regionId() ) )
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
