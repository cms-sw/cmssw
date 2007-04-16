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
// $Id: L1ExtraParticleMapProd.cc,v 1.8 2007/04/13 17:50:48 wsun Exp $
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

   singleThresholds_[ L1ParticleMap::kSingleMu3 ] =
      iConfig.getParameter< double >( "A_SingleMu3_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu3 ] =
      iConfig.getParameter< int >( "A_SingleMu3_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu10 ] =
      iConfig.getParameter< double >( "A_SingleMu10_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu10 ] =
      iConfig.getParameter< int >( "A_SingleMu10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu14 ] =
      iConfig.getParameter< double >( "A_SingleMu14_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu14 ] =
      iConfig.getParameter< int >( "A_SingleMu14_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu20 ] =
      iConfig.getParameter< double >( "A_SingleMu20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu20 ] =
      iConfig.getParameter< int >( "A_SingleMu20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu25 ] =
      iConfig.getParameter< double >( "A_SingleMu25_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu25 ] =
      iConfig.getParameter< int >( "A_SingleMu25_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kSingleIsoEG5 ] =
      iConfig.getParameter< double >( "A_SingleIsoEG5_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG5 ] =
      iConfig.getParameter< int >( "A_SingleIsoEG5_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG10 ] =
      iConfig.getParameter< double >( "A_SingleIsoEG10_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG10 ] =
      iConfig.getParameter< int >( "A_SingleIsoEG10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG15 ] =
      iConfig.getParameter< double >( "A_SingleIsoEG15_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG15 ] =
      iConfig.getParameter< int >( "A_SingleIsoEG15_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG20 ] =
      iConfig.getParameter< double >( "A_SingleIsoEG20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG20 ] =
      iConfig.getParameter< int >( "A_SingleIsoEG20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG25 ] =
      iConfig.getParameter< double >( "A_SingleIsoEG25_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG25 ] =
      iConfig.getParameter< int >( "A_SingleIsoEG25_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kSingleEG5 ] =
      iConfig.getParameter< double >( "A_SingleEG5_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG5 ] =
      iConfig.getParameter< int >( "A_SingleEG5_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG10 ] =
      iConfig.getParameter< double >( "A_SingleEG10_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG10 ] =
      iConfig.getParameter< int >( "A_SingleEG10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG15 ] =
      iConfig.getParameter< double >( "A_SingleEG15_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG15 ] =
      iConfig.getParameter< int >( "A_SingleEG15_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG20 ] =
      iConfig.getParameter< double >( "A_SingleEG20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG20 ] =
      iConfig.getParameter< int >( "A_SingleEG20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG25 ] =
      iConfig.getParameter< double >( "A_SingleEG25_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG25 ] =
      iConfig.getParameter< int >( "A_SingleEG25_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kSingleJet20 ] =
      iConfig.getParameter< double >( "A_SingleJet20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet20 ] =
      iConfig.getParameter< int >( "A_SingleJet20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet60 ] =
      iConfig.getParameter< double >( "A_SingleJet60_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet60 ] =
      iConfig.getParameter< int >( "A_SingleJet60_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet100 ] =
      iConfig.getParameter< double >( "A_SingleJet100_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet100 ] =
      iConfig.getParameter< int >( "A_SingleJet100_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet140 ] =
      iConfig.getParameter< double >( "A_SingleJet140_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet140 ] =
      iConfig.getParameter< int >( "A_SingleJet140_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet180 ] =
      iConfig.getParameter< double >( "A_SingleJet180_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet180 ] =
      iConfig.getParameter< int >( "A_SingleJet180_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kSingleTauJet20 ] =
      iConfig.getParameter< double >( "A_SingleTauJet20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet20 ] =
      iConfig.getParameter< int >( "A_SingleTauJet20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet60 ] =
      iConfig.getParameter< double >( "A_SingleTauJet60_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet60 ] =
      iConfig.getParameter< int >( "A_SingleTauJet60_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet100 ] =
      iConfig.getParameter< double >( "A_SingleTauJet100_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet100 ] =
      iConfig.getParameter< int >( "A_SingleTauJet100_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet140 ] =
      iConfig.getParameter< double >( "A_SingleTauJet140_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet140 ] =
      iConfig.getParameter< int >( "A_SingleTauJet140_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet180 ] =
      iConfig.getParameter< double >( "A_SingleTauJet180_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet180 ] =
      iConfig.getParameter< int >( "A_SingleTauJet180_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kHTT100 ] =
      iConfig.getParameter< double >( "A_HTT100_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT100 ] =
      iConfig.getParameter< int >( "A_HTT100_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT200 ] =
      iConfig.getParameter< double >( "A_HTT200_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT200 ] =
      iConfig.getParameter< int >( "A_HTT200_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT300 ] =
      iConfig.getParameter< double >( "A_HTT300_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT300 ] =
      iConfig.getParameter< int >( "A_HTT300_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT400 ] =
      iConfig.getParameter< double >( "A_HTT400_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT400 ] =
      iConfig.getParameter< int >( "A_HTT400_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT500 ] =
      iConfig.getParameter< double >( "A_HTT500_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT500 ] =
      iConfig.getParameter< int >( "A_HTT500_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kETM20 ] =
      iConfig.getParameter< double >( "A_ETM20_thresh" ) ;
   prescales_[ L1ParticleMap::kETM20 ] =
      iConfig.getParameter< int >( "A_ETM20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM40 ] =
      iConfig.getParameter< double >( "A_ETM40_thresh" ) ;
   prescales_[ L1ParticleMap::kETM40 ] =
      iConfig.getParameter< int >( "A_ETM40_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM60 ] =
      iConfig.getParameter< double >( "A_ETM60_thresh" ) ;
   prescales_[ L1ParticleMap::kETM60 ] =
      iConfig.getParameter< int >( "A_ETM60_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM80 ] =
      iConfig.getParameter< double >( "A_ETM80_thresh" ) ;
   prescales_[ L1ParticleMap::kETM80 ] =
      iConfig.getParameter< int >( "A_ETM80_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM100 ] =
      iConfig.getParameter< double >( "A_ETM100_thresh" ) ;
   prescales_[ L1ParticleMap::kETM100 ] =
      iConfig.getParameter< int >( "A_ETM100_prescale" ) ;

   // AA triggers

   singleThresholds_[ L1ParticleMap::kDoubleMu7 ] =
      iConfig.getParameter< double >( "A_DoubleMu7_thresh" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleIsoEG10 ] =
      iConfig.getParameter< double >( "A_DoubleIsoEG10_thresh" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleEG20 ] =
      iConfig.getParameter< double >( "A_DoubleEG20_thresh" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleJet100 ] =
      iConfig.getParameter< double >( "A_DoubleJet100_thresh" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleTauJet50 ] =
      iConfig.getParameter< double >( "A_DoubleTauJet50_thresh" ) ;

   // AB triggers

   doubleThresholds_[ L1ParticleMap::kMu7_IsoEG15 ].first =
      iConfig.getParameter< double >( "A_Mu7_IsoEG15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_IsoEG15 ].second =
      iConfig.getParameter< double >( "A_Mu7_IsoEG15_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_EG15 ].first =
      iConfig.getParameter< double >( "A_Mu7_EG15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_EG15 ].second =
      iConfig.getParameter< double >( "A_Mu7_EG15_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_Jet100 ].first =
      iConfig.getParameter< double >( "A_Mu7_Jet100_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_Jet100 ].second =
      iConfig.getParameter< double >( "A_Mu7_Jet100_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_TauJet50 ].first =
      iConfig.getParameter< double >( "A_Mu7_TauJet50_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_TauJet50 ].second =
      iConfig.getParameter< double >( "A_Mu7_TauJet50_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_EG10 ].first =
      iConfig.getParameter< double >( "A_IsoEG10_EG10_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_EG10 ].second =
      iConfig.getParameter< double >( "A_IsoEG10_EG10_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG15_Jet100 ].first =
      iConfig.getParameter< double >( "A_IsoEG15_Jet100_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG15_Jet100 ].second =
      iConfig.getParameter< double >( "A_IsoEG15_Jet100_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG15_TauJet50 ].first =
      iConfig.getParameter< double >( "A_IsoEG15_TauJet50_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG15_TauJet50 ].second =
      iConfig.getParameter< double >( "A_IsoEG15_TauJet50_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kEG15_Jet100 ].first =
      iConfig.getParameter< double >( "A_EG15_Jet100_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG15_Jet100 ].second =
      iConfig.getParameter< double >( "A_EG15_Jet100_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kEG15_TauJet50 ].first =
      iConfig.getParameter< double >( "A_EG15_TauJet50_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG15_TauJet50 ].second =
      iConfig.getParameter< double >( "A_EG15_TauJet50_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kJet100_TauJet50 ].first =
      iConfig.getParameter< double >( "A_Jet100_TauJet50_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJet100_TauJet50 ].second =
      iConfig.getParameter< double >( "A_Jet100_TauJet50_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_HTT300 ].first =
      iConfig.getParameter< double >( "A_Mu7_HTT300_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_HTT300 ].second =
      iConfig.getParameter< double >( "A_Mu7_HTT300_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG15_HTT300 ].first =
      iConfig.getParameter< double >( "A_IsoEG15_HTT300_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG15_HTT300 ].second =
      iConfig.getParameter< double >( "A_IsoEG15_HTT300_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kEG15_HTT300 ].first =
      iConfig.getParameter< double >( "A_EG15_HTT300_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG15_HTT300 ].second =
      iConfig.getParameter< double >( "A_EG15_HTT300_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kJet100_HTT300 ].first =
      iConfig.getParameter< double >( "A_Jet100_HTT300_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJet100_HTT300 ].second =
      iConfig.getParameter< double >( "A_Jet100_HTT300_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet50_HTT300 ].first =
      iConfig.getParameter< double >( "A_TauJet50_HTT300_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet50_HTT300 ].second =
      iConfig.getParameter< double >( "A_TauJet50_HTT300_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_ETM30 ].first =
      iConfig.getParameter< double >( "A_Mu7_ETM30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu7_ETM30 ].second =
      iConfig.getParameter< double >( "A_Mu7_ETM30_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG15_ETM30 ].first =
      iConfig.getParameter< double >( "A_IsoEG15_ETM30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG15_ETM30 ].second =
      iConfig.getParameter< double >( "A_IsoEG15_ETM30_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kEG15_ETM30 ].first =
      iConfig.getParameter< double >( "A_EG15_ETM30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG15_ETM30 ].second =
      iConfig.getParameter< double >( "A_EG15_ETM30_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kJet100_ETM40 ].first =
      iConfig.getParameter< double >( "A_Jet100_ETM40_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJet100_ETM40 ].second =
      iConfig.getParameter< double >( "A_Jet100_ETM40_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet50_ETM40 ].first =
      iConfig.getParameter< double >( "A_TauJet50_ETM40_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet50_ETM40 ].second =
      iConfig.getParameter< double >( "A_TauJet50_ETM40_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kHTT200_ETM40 ].first =
      iConfig.getParameter< double >( "A_HTT200_ETM40_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kHTT200_ETM40 ].second =
      iConfig.getParameter< double >( "A_HTT200_ETM40_thresh2" ) ;

   // AAA triggers

   singleThresholds_[ L1ParticleMap::kTripleMu7 ] =
      iConfig.getParameter< double >( "A_TripleMu7_thresh" ) ;
   singleThresholds_[ L1ParticleMap::kTripleIsoEG5 ] =
      iConfig.getParameter< double >( "A_TripleIsoEG5_thresh" ) ;
   singleThresholds_[ L1ParticleMap::kTripleEG10 ] =
      iConfig.getParameter< double >( "A_TripleEG10_thresh" ) ;
   singleThresholds_[ L1ParticleMap::kTripleJet70 ] =
      iConfig.getParameter< double >( "A_TripleJet70_thresh" ) ;
   singleThresholds_[ L1ParticleMap::kTripleTauJet50 ] =
      iConfig.getParameter< double >( "A_TripleTauJet50_thresh" ) ;

   // AAB triggers

   doubleThresholds_[ L1ParticleMap::kDoubleMu7_IsoEG5 ].first =
      iConfig.getParameter< double >( "A_DoubleMu7_IsoEG5_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu7_IsoEG5 ].second =
      iConfig.getParameter< double >( "A_DoubleMu7_IsoEG5_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu7_EG5 ].first =
      iConfig.getParameter< double >( "A_DoubleMu7_EG5_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu7_EG5 ].second =
      iConfig.getParameter< double >( "A_DoubleMu7_EG5_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_Mu7 ].first =
      iConfig.getParameter< double >( "A_DoubleIsoEG5_Mu7_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_Mu7 ].second =
      iConfig.getParameter< double >( "A_DoubleIsoEG5_Mu7_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG5_Mu7 ].first =
      iConfig.getParameter< double >( "A_DoubleEG5_Mu7_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG5_Mu7 ].second =
      iConfig.getParameter< double >( "A_DoubleEG5_Mu7_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu7_HTT150 ].first =
      iConfig.getParameter< double >( "A_DoubleMu7_HTT150_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu7_HTT150 ].second =
      iConfig.getParameter< double >( "A_DoubleMu7_HTT150_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_HTT150 ].first =
      iConfig.getParameter< double >( "A_DoubleIsoEG5_HTT150_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_HTT150 ].second =
      iConfig.getParameter< double >( "A_DoubleIsoEG5_HTT150_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG5_HTT150 ].first =
      iConfig.getParameter< double >( "A_DoubleEG5_HTT150_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG5_HTT150 ].second =
      iConfig.getParameter< double >( "A_DoubleEG5_HTT150_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJet50_HTT150 ].first =
      iConfig.getParameter< double >( "A_DoubleJet50_HTT150_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJet50_HTT150 ].second =
      iConfig.getParameter< double >( "A_DoubleJet50_HTT150_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauJet50_HTT150 ].first =
      iConfig.getParameter< double >( "A_DoubleTauJet50_HTT150_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauJet50_HTT150 ].second =
      iConfig.getParameter< double >( "A_DoubleTauJet50_HTT150_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu7_ETM15 ].first =
      iConfig.getParameter< double >( "A_DoubleMu7_ETM15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu7_ETM15 ].second =
      iConfig.getParameter< double >( "A_DoubleMu7_ETM15_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_ETM15 ].first =
      iConfig.getParameter< double >( "A_DoubleIsoEG5_ETM15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_ETM15 ].second =
      iConfig.getParameter< double >( "A_DoubleIsoEG5_ETM15_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG5_ETM15 ].first =
      iConfig.getParameter< double >( "A_DoubleEG5_ETM15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG5_ETM15 ].second =
      iConfig.getParameter< double >( "A_DoubleEG5_ETM15_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJet50_ETM15 ].first =
      iConfig.getParameter< double >( "A_DoubleJet50_ETM15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJet50_ETM15 ].second =
      iConfig.getParameter< double >( "A_DoubleJet50_ETM15_thresh2" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauJet50_ETM15 ].first =
      iConfig.getParameter< double >( "A_DoubleTauJet50_ETM15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauJet50_ETM15 ].second =
      iConfig.getParameter< double >( "A_DoubleTauJet50_ETM15_thresh2" ) ;

   singleThresholds_[ L1ParticleMap::kQuadJet50 ] =
      iConfig.getParameter< double >( "A_QuadJet50_thresh" ) ;

   prescales_[ L1ParticleMap::kMinBias_HTT10 ] =
      iConfig.getParameter< int >( "A_MinBias_HTT10_prescale" ) ;
   prescales_[ L1ParticleMap::kZeroBias ] =
      iConfig.getParameter< int >( "A_ZeroBias_prescale" ) ;

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

      if( itrig >= L1ParticleMap::kSingleMu3 &&
	  itrig <= L1ParticleMap::kSingleMu25 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateSingleObjectTrigger( inputMuonRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputMuonRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kSingleIsoEG5 &&
	       itrig <= L1ParticleMap::kSingleIsoEG25 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateSingleObjectTrigger( inputIsoEmRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputEmRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kSingleEG5 &&
	       itrig <= L1ParticleMap::kSingleEG25 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateSingleObjectTrigger( inputRelaxedEmRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputEmRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kSingleJet20 &&
	       itrig <= L1ParticleMap::kSingleJet180 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateSingleObjectTrigger( inputJetRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputJetRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kSingleTauJet20 &&
	       itrig <= L1ParticleMap::kSingleTauJet180 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateSingleObjectTrigger( inputTauRefs,
				      singleThresholds_[ itrig ],
				      prescaleCounters_[ itrig ],
				      prescales_[ itrig ],
				      decision,
				      outputJetRefs ) ;
      }
      else if( itrig >= L1ParticleMap::kHTT100 &&
	       itrig <= L1ParticleMap::kHTT500 )
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
      else if( itrig >= L1ParticleMap::kETM20 &&
	       itrig <= L1ParticleMap::kETM100 )
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
      else if( itrig == L1ParticleMap::kDoubleMu7 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoubleSameObjectTrigger( inputMuonRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputMuonRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEG10 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleSameObjectTrigger( inputIsoEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleEG20 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleSameObjectTrigger( inputRelaxedEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleJet100 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleSameObjectTrigger( inputJetRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleTauJet50 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleSameObjectTrigger( inputTauRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefs,
					  combos ) ;
      }
      else if( itrig == L1ParticleMap::kMu7_IsoEG15 )
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
      else if( itrig == L1ParticleMap::kMu7_EG15 )
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
      else if( itrig == L1ParticleMap::kMu7_Jet100 )
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
      else if( itrig == L1ParticleMap::kMu7_TauJet50 )
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
      else if( itrig == L1ParticleMap::kIsoEG10_EG10 )
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
      else if( itrig == L1ParticleMap::kIsoEG15_Jet100 )
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
      else if( itrig == L1ParticleMap::kIsoEG15_TauJet50 )
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
      else if( itrig == L1ParticleMap::kEG15_Jet100 )
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
      else if( itrig == L1ParticleMap::kEG15_TauJet50 )
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
      else if( itrig == L1ParticleMap::kJet100_TauJet50 )
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
      else if( itrig == L1ParticleMap::kMu7_HTT300 )
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
      else if( itrig == L1ParticleMap::kIsoEG15_HTT300 )
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
      else if( itrig == L1ParticleMap::kEG15_HTT300 )
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
      else if( itrig == L1ParticleMap::kJet100_HTT300 )
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
      else if( itrig == L1ParticleMap::kTauJet50_HTT300 )
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
      else if( itrig == L1ParticleMap::kMu7_ETM30 )
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
      else if( itrig == L1ParticleMap::kIsoEG15_ETM30 )
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
      else if( itrig == L1ParticleMap::kEG15_ETM30 )
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
      else if( itrig == L1ParticleMap::kJet100_ETM40 )
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
      else if( itrig == L1ParticleMap::kTauJet50_ETM40 )
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
      else if( itrig == L1ParticleMap::kHTT200_ETM40 )
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
      else if( itrig == L1ParticleMap::kTripleMu7 )
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
      else if( itrig == L1ParticleMap::kTripleIsoEG5 )
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
      else if( itrig == L1ParticleMap::kTripleEG10 )
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
      else if( itrig == L1ParticleMap::kTripleJet70 )
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
      else if( itrig == L1ParticleMap::kTripleTauJet50 )
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
      else if( itrig == L1ParticleMap::kDoubleMu7_IsoEG5 )
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
      else if( itrig == L1ParticleMap::kDoubleMu7_EG5 )
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
      else if( itrig == L1ParticleMap::kDoubleIsoEG5_Mu7 )
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
      else if( itrig == L1ParticleMap::kDoubleEG5_Mu7 )
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
      else if( itrig == L1ParticleMap::kDoubleMu7_HTT150 )
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
      else if( itrig == L1ParticleMap::kDoubleIsoEG5_HTT150 )
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
      else if( itrig == L1ParticleMap::kDoubleEG5_HTT150 )
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
      else if( itrig == L1ParticleMap::kDoubleJet50_HTT150 )
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
      else if( itrig == L1ParticleMap::kDoubleTauJet50_HTT150 )
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
      else if( itrig == L1ParticleMap::kDoubleMu7_ETM15 )
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
      else if( itrig == L1ParticleMap::kDoubleIsoEG5_ETM15 )
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
      else if( itrig == L1ParticleMap::kDoubleEG5_ETM15 )
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
      else if( itrig == L1ParticleMap::kDoubleJet50_ETM15 )
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
      else if( itrig == L1ParticleMap::kDoubleTauJet50_ETM15 )
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
      else if( itrig == L1ParticleMap::kQuadJet50 )
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


      else if( itrig == L1ParticleMap::kMinBias_HTT10 )
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
