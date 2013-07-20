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
// $Id: L1ExtraParticleMapProd.cc,v 1.30 2009/05/27 11:12:11 fabiocos Exp $
//
//


// system include files
#include <memory>

// user include files
#include "L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticleMapProd.h"

//#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandFlat.h"

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
        "etMissSource" ) ),
     htMissSource_( iConfig.getParameter< edm::InputTag >(
	"htMissSource" ) )
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
      prescaleCounters_[ i ] = 1 ;
      prescales_[ i ] = 1 ;
   }

   // Single object triggers, 5 thresholds each

   singleThresholds_[ L1ParticleMap::kSingleMu3 ] =
      iConfig.getParameter< double >( "L1_SingleMu3_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu3 ] =
      iConfig.getParameter< int >( "L1_SingleMu3_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu5 ] =
      iConfig.getParameter< double >( "L1_SingleMu5_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu5 ] =
      iConfig.getParameter< int >( "L1_SingleMu5_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu7 ] =
      iConfig.getParameter< double >( "L1_SingleMu7_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu7 ] =
      iConfig.getParameter< int >( "L1_SingleMu7_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu10 ] =
      iConfig.getParameter< double >( "L1_SingleMu10_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu10 ] =
      iConfig.getParameter< int >( "L1_SingleMu10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu14 ] =
      iConfig.getParameter< double >( "L1_SingleMu14_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu14 ] =
      iConfig.getParameter< int >( "L1_SingleMu14_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu20 ] =
      iConfig.getParameter< double >( "L1_SingleMu20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu20 ] =
      iConfig.getParameter< int >( "L1_SingleMu20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleMu25 ] =
      iConfig.getParameter< double >( "L1_SingleMu25_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleMu25 ] =
      iConfig.getParameter< int >( "L1_SingleMu25_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kSingleIsoEG5 ] =
      iConfig.getParameter< double >( "L1_SingleIsoEG5_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG5 ] =
      iConfig.getParameter< int >( "L1_SingleIsoEG5_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG8 ] =
      iConfig.getParameter< double >( "L1_SingleIsoEG8_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG8 ] =
      iConfig.getParameter< int >( "L1_SingleIsoEG8_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG10 ] =
      iConfig.getParameter< double >( "L1_SingleIsoEG10_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG10 ] =
      iConfig.getParameter< int >( "L1_SingleIsoEG10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG12 ] =
      iConfig.getParameter< double >( "L1_SingleIsoEG12_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG12 ] =
      iConfig.getParameter< int >( "L1_SingleIsoEG12_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG15 ] =
      iConfig.getParameter< double >( "L1_SingleIsoEG15_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG15 ] =
      iConfig.getParameter< int >( "L1_SingleIsoEG15_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG20 ] =
      iConfig.getParameter< double >( "L1_SingleIsoEG20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG20 ] =
      iConfig.getParameter< int >( "L1_SingleIsoEG20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleIsoEG25 ] =
      iConfig.getParameter< double >( "L1_SingleIsoEG25_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleIsoEG25 ] =
      iConfig.getParameter< int >( "L1_SingleIsoEG25_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kSingleEG5 ] =
      iConfig.getParameter< double >( "L1_SingleEG5_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG5 ] =
      iConfig.getParameter< int >( "L1_SingleEG5_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG8 ] =
      iConfig.getParameter< double >( "L1_SingleEG8_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG8 ] =
      iConfig.getParameter< int >( "L1_SingleEG8_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG10 ] =
      iConfig.getParameter< double >( "L1_SingleEG10_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG10 ] =
      iConfig.getParameter< int >( "L1_SingleEG10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG12 ] =
      iConfig.getParameter< double >( "L1_SingleEG12_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG12 ] =
      iConfig.getParameter< int >( "L1_SingleEG12_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG15 ] =
      iConfig.getParameter< double >( "L1_SingleEG15_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG15 ] =
      iConfig.getParameter< int >( "L1_SingleEG15_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG20 ] =
      iConfig.getParameter< double >( "L1_SingleEG20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG20 ] =
      iConfig.getParameter< int >( "L1_SingleEG20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleEG25 ] =
      iConfig.getParameter< double >( "L1_SingleEG25_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleEG25 ] =
      iConfig.getParameter< int >( "L1_SingleEG25_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kSingleJet15 ] =
      iConfig.getParameter< double >( "L1_SingleJet15_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet15 ] =
      iConfig.getParameter< int >( "L1_SingleJet15_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet20 ] =
      iConfig.getParameter< double >( "L1_SingleJet20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet20 ] =
      iConfig.getParameter< int >( "L1_SingleJet20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet30 ] =
      iConfig.getParameter< double >( "L1_SingleJet30_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet30 ] =
      iConfig.getParameter< int >( "L1_SingleJet30_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet50 ] =
      iConfig.getParameter< double >( "L1_SingleJet50_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet50 ] =
      iConfig.getParameter< int >( "L1_SingleJet50_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet70 ] =
      iConfig.getParameter< double >( "L1_SingleJet70_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet70 ] =
      iConfig.getParameter< int >( "L1_SingleJet70_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet100 ] =
      iConfig.getParameter< double >( "L1_SingleJet100_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet100 ] =
      iConfig.getParameter< int >( "L1_SingleJet100_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet150 ] =
      iConfig.getParameter< double >( "L1_SingleJet150_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet150 ] =
      iConfig.getParameter< int >( "L1_SingleJet150_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleJet200 ] =
      iConfig.getParameter< double >( "L1_SingleJet200_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleJet200 ] =
      iConfig.getParameter< int >( "L1_SingleJet200_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kSingleTauJet10 ] =
      iConfig.getParameter< double >( "L1_SingleTauJet10_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet10 ] =
      iConfig.getParameter< int >( "L1_SingleTauJet10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet20 ] =
      iConfig.getParameter< double >( "L1_SingleTauJet20_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet20 ] =
      iConfig.getParameter< int >( "L1_SingleTauJet20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet30 ] =
      iConfig.getParameter< double >( "L1_SingleTauJet30_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet30 ] =
      iConfig.getParameter< int >( "L1_SingleTauJet30_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet35 ] =
      iConfig.getParameter< double >( "L1_SingleTauJet35_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet35 ] =
      iConfig.getParameter< int >( "L1_SingleTauJet35_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet40 ] =
      iConfig.getParameter< double >( "L1_SingleTauJet40_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet40 ] =
      iConfig.getParameter< int >( "L1_SingleTauJet40_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet60 ] =
      iConfig.getParameter< double >( "L1_SingleTauJet60_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet60 ] =
      iConfig.getParameter< int >( "L1_SingleTauJet60_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet80 ] =
      iConfig.getParameter< double >( "L1_SingleTauJet80_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet80 ] =
      iConfig.getParameter< int >( "L1_SingleTauJet80_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kSingleTauJet100 ] =
      iConfig.getParameter< double >( "L1_SingleTauJet100_thresh" ) ;
   prescales_[ L1ParticleMap::kSingleTauJet100 ] =
      iConfig.getParameter< int >( "L1_SingleTauJet100_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kHTT100 ] =
      iConfig.getParameter< double >( "L1_HTT100_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT100 ] =
      iConfig.getParameter< int >( "L1_HTT100_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT200 ] =
      iConfig.getParameter< double >( "L1_HTT200_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT200 ] =
      iConfig.getParameter< int >( "L1_HTT200_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT250 ] =
      iConfig.getParameter< double >( "L1_HTT250_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT250 ] =
      iConfig.getParameter< int >( "L1_HTT250_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT300 ] =
      iConfig.getParameter< double >( "L1_HTT300_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT300 ] =
      iConfig.getParameter< int >( "L1_HTT300_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT400 ] =
      iConfig.getParameter< double >( "L1_HTT400_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT400 ] =
      iConfig.getParameter< int >( "L1_HTT400_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kHTT500 ] =
      iConfig.getParameter< double >( "L1_HTT500_thresh" ) ;
   prescales_[ L1ParticleMap::kHTT500 ] =
      iConfig.getParameter< int >( "L1_HTT500_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kETM10 ] =
      iConfig.getParameter< double >( "L1_ETM10_thresh" ) ;
   prescales_[ L1ParticleMap::kETM10 ] =
      iConfig.getParameter< int >( "L1_ETM10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM15 ] =
      iConfig.getParameter< double >( "L1_ETM15_thresh" ) ;
   prescales_[ L1ParticleMap::kETM15 ] =
      iConfig.getParameter< int >( "L1_ETM15_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM20 ] =
      iConfig.getParameter< double >( "L1_ETM20_thresh" ) ;
   prescales_[ L1ParticleMap::kETM20 ] =
      iConfig.getParameter< int >( "L1_ETM20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM30 ] =
      iConfig.getParameter< double >( "L1_ETM30_thresh" ) ;
   prescales_[ L1ParticleMap::kETM30 ] =
      iConfig.getParameter< int >( "L1_ETM30_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM40 ] =
      iConfig.getParameter< double >( "L1_ETM40_thresh" ) ;
   prescales_[ L1ParticleMap::kETM40 ] =
      iConfig.getParameter< int >( "L1_ETM40_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM50 ] =
      iConfig.getParameter< double >( "L1_ETM50_thresh" ) ;
   prescales_[ L1ParticleMap::kETM50 ] =
      iConfig.getParameter< int >( "L1_ETM50_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kETM60 ] =
      iConfig.getParameter< double >( "L1_ETM60_thresh" ) ;
   prescales_[ L1ParticleMap::kETM60 ] =
      iConfig.getParameter< int >( "L1_ETM60_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kETT60 ] =
      iConfig.getParameter< double >( "L1_ETT60_thresh" ) ;
   prescales_[ L1ParticleMap::kETT60 ] =
      iConfig.getParameter< int >( "L1_ETT60_prescale" ) ;

   // AA triggers

   singleThresholds_[ L1ParticleMap::kDoubleMu3 ] =
      iConfig.getParameter< double >( "L1_DoubleMu3_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleMu3 ] =
      iConfig.getParameter< int >( "L1_DoubleMu3_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kDoubleIsoEG8 ] =
      iConfig.getParameter< double >( "L1_DoubleIsoEG8_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleIsoEG8 ] =
      iConfig.getParameter< int >( "L1_DoubleIsoEG8_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleIsoEG10 ] =
      iConfig.getParameter< double >( "L1_DoubleIsoEG10_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleIsoEG10 ] =
      iConfig.getParameter< int >( "L1_DoubleIsoEG10_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kDoubleEG5 ] =
      iConfig.getParameter< double >( "L1_DoubleEG5_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleEG5 ] =
      iConfig.getParameter< int >( "L1_DoubleEG5_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleEG10 ] =
      iConfig.getParameter< double >( "L1_DoubleEG10_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleEG10 ] =
      iConfig.getParameter< int >( "L1_DoubleEG10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleEG15 ] =
      iConfig.getParameter< double >( "L1_DoubleEG15_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleEG15 ] =
      iConfig.getParameter< int >( "L1_DoubleEG15_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kDoubleJet70 ] =
      iConfig.getParameter< double >( "L1_DoubleJet70_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleJet70 ] =
      iConfig.getParameter< int >( "L1_DoubleJet70_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleJet100 ] =
      iConfig.getParameter< double >( "L1_DoubleJet100_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleJet100 ] =
      iConfig.getParameter< int >( "L1_DoubleJet100_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kDoubleTauJet20 ] =
      iConfig.getParameter< double >( "L1_DoubleTauJet20_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleTauJet20 ] =
      iConfig.getParameter< int >( "L1_DoubleTauJet20_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleTauJet30 ] =
      iConfig.getParameter< double >( "L1_DoubleTauJet30_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleTauJet30 ] =
      iConfig.getParameter< int >( "L1_DoubleTauJet30_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleTauJet35 ] =
      iConfig.getParameter< double >( "L1_DoubleTauJet35_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleTauJet35 ] =
      iConfig.getParameter< int >( "L1_DoubleTauJet35_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kDoubleTauJet40 ] =
      iConfig.getParameter< double >( "L1_DoubleTauJet40_thresh" ) ;
   prescales_[ L1ParticleMap::kDoubleTauJet40 ] =
      iConfig.getParameter< int >( "L1_DoubleTauJet40_prescale" ) ;

   // AB triggers

   doubleThresholds_[ L1ParticleMap::kMu3_IsoEG5 ].first =
      iConfig.getParameter< double >( "L1_Mu3_IsoEG5_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu3_IsoEG5 ].second =
      iConfig.getParameter< double >( "L1_Mu3_IsoEG5_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu3_IsoEG5 ] =
      iConfig.getParameter< int >( "L1_Mu3_IsoEG5_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_IsoEG10 ].first =
      iConfig.getParameter< double >( "L1_Mu5_IsoEG10_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_IsoEG10 ].second =
      iConfig.getParameter< double >( "L1_Mu5_IsoEG10_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu5_IsoEG10 ] =
      iConfig.getParameter< int >( "L1_Mu5_IsoEG10_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kMu3_EG12 ].first =
      iConfig.getParameter< double >( "L1_Mu3_EG12_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu3_EG12 ].second =
      iConfig.getParameter< double >( "L1_Mu3_EG12_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu3_EG12 ] =
      iConfig.getParameter< int >( "L1_Mu3_EG12_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kMu3_Jet15 ].first =
      iConfig.getParameter< double >( "L1_Mu3_Jet15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu3_Jet15 ].second =
      iConfig.getParameter< double >( "L1_Mu3_Jet15_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu3_Jet15 ] =
      iConfig.getParameter< int >( "L1_Mu3_Jet15_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_Jet15 ].first =
      iConfig.getParameter< double >( "L1_Mu5_Jet15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_Jet15 ].second =
      iConfig.getParameter< double >( "L1_Mu5_Jet15_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu5_Jet15 ] =
      iConfig.getParameter< int >( "L1_Mu5_Jet15_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kMu3_Jet70 ].first =
      iConfig.getParameter< double >( "L1_Mu3_Jet70_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu3_Jet70 ].second =
      iConfig.getParameter< double >( "L1_Mu3_Jet70_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu3_Jet70 ] =
      iConfig.getParameter< int >( "L1_Mu3_Jet70_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_Jet20 ].first =
      iConfig.getParameter< double >( "L1_Mu5_Jet20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_Jet20 ].second =
      iConfig.getParameter< double >( "L1_Mu5_Jet20_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu5_Jet20 ] =
      iConfig.getParameter< int >( "L1_Mu5_Jet20_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kMu5_TauJet20 ].first =
      iConfig.getParameter< double >( "L1_Mu5_TauJet20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_TauJet20 ].second =
      iConfig.getParameter< double >( "L1_Mu5_TauJet20_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu5_TauJet20 ] =
      iConfig.getParameter< int >( "L1_Mu5_TauJet20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_TauJet30 ].first =
      iConfig.getParameter< double >( "L1_Mu5_TauJet30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu5_TauJet30 ].second =
      iConfig.getParameter< double >( "L1_Mu5_TauJet30_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu5_TauJet30 ] =
      iConfig.getParameter< int >( "L1_Mu5_TauJet30_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kIsoEG10_EG10 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_EG10_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_EG10 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_EG10_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_EG10 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_EG10_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet15 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet15 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet15_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_Jet15 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_Jet15_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet30 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet30 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet30_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_Jet30 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_Jet30_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet20 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet20 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet20_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_Jet20 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_Jet20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet70 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet70_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet70 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet70_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_Jet70 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_Jet70_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kIsoEG10_TauJet20 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_TauJet20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_TauJet20 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_TauJet20_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_TauJet20 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_TauJet20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_TauJet30 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_TauJet30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_TauJet30 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_TauJet30_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_TauJet30 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_TauJet30_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kEG10_Jet15 ].first =
      iConfig.getParameter< double >( "L1_EG10_Jet15_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG10_Jet15 ].second =
      iConfig.getParameter< double >( "L1_EG10_Jet15_thresh2" ) ;
   prescales_[ L1ParticleMap::kEG10_Jet15 ] =
      iConfig.getParameter< int >( "L1_EG10_Jet15_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_Jet20 ].first =
      iConfig.getParameter< double >( "L1_EG12_Jet20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_Jet20 ].second =
      iConfig.getParameter< double >( "L1_EG12_Jet20_thresh2" ) ;
   prescales_[ L1ParticleMap::kEG12_Jet20 ] =
      iConfig.getParameter< int >( "L1_EG12_Jet20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_Jet70 ].first =
      iConfig.getParameter< double >( "L1_EG12_Jet70_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_Jet70 ].second =
      iConfig.getParameter< double >( "L1_EG12_Jet70_thresh2" ) ;
   prescales_[ L1ParticleMap::kEG12_Jet70 ] =
      iConfig.getParameter< int >( "L1_EG12_Jet70_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kEG12_TauJet40 ].first =
      iConfig.getParameter< double >( "L1_EG12_TauJet40_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_TauJet40 ].second =
      iConfig.getParameter< double >( "L1_EG12_TauJet40_thresh2" ) ;
   prescales_[ L1ParticleMap::kEG12_TauJet40 ] =
      iConfig.getParameter< int >( "L1_EG12_TauJet40_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kJet70_TauJet40 ].first =
      iConfig.getParameter< double >( "L1_Jet70_TauJet40_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJet70_TauJet40 ].second =
      iConfig.getParameter< double >( "L1_Jet70_TauJet40_thresh2" ) ;
   prescales_[ L1ParticleMap::kJet70_TauJet40 ] =
      iConfig.getParameter< int >( "L1_Jet70_TauJet40_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kMu3_HTT200 ].first =
      iConfig.getParameter< double >( "L1_Mu3_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu3_HTT200 ].second =
      iConfig.getParameter< double >( "L1_Mu3_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu3_HTT200 ] =
      iConfig.getParameter< int >( "L1_Mu3_HTT200_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_HTT200 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_HTT200 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_HTT200 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_HTT200_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_HTT200 ].first =
      iConfig.getParameter< double >( "L1_EG12_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_HTT200 ].second =
      iConfig.getParameter< double >( "L1_EG12_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kEG12_HTT200 ] =
      iConfig.getParameter< int >( "L1_EG12_HTT200_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kJet70_HTT200 ].first =
      iConfig.getParameter< double >( "L1_Jet70_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJet70_HTT200 ].second =
      iConfig.getParameter< double >( "L1_Jet70_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kJet70_HTT200 ] =
      iConfig.getParameter< int >( "L1_Jet70_HTT200_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet40_HTT200 ].first =
      iConfig.getParameter< double >( "L1_TauJet40_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet40_HTT200 ].second =
      iConfig.getParameter< double >( "L1_TauJet40_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kTauJet40_HTT200 ] =
      iConfig.getParameter< int >( "L1_TauJet40_HTT200_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kMu3_ETM30 ].first =
      iConfig.getParameter< double >( "L1_Mu3_ETM30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kMu3_ETM30 ].second =
      iConfig.getParameter< double >( "L1_Mu3_ETM30_thresh2" ) ;
   prescales_[ L1ParticleMap::kMu3_ETM30 ] =
      iConfig.getParameter< int >( "L1_Mu3_ETM30_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_ETM30 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_ETM30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_ETM30 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_ETM30_thresh2" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_ETM30 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_ETM30_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_ETM30 ].first =
      iConfig.getParameter< double >( "L1_EG12_ETM30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kEG12_ETM30 ].second =
      iConfig.getParameter< double >( "L1_EG12_ETM30_thresh2" ) ;
   prescales_[ L1ParticleMap::kEG12_ETM30 ] =
      iConfig.getParameter< int >( "L1_EG12_ETM30_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kJet70_ETM40 ].first =
      iConfig.getParameter< double >( "L1_Jet70_ETM40_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kJet70_ETM40 ].second =
      iConfig.getParameter< double >( "L1_Jet70_ETM40_thresh2" ) ;
   prescales_[ L1ParticleMap::kJet70_ETM40 ] =
      iConfig.getParameter< int >( "L1_Jet70_ETM40_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kTauJet20_ETM20 ].first =
      iConfig.getParameter< double >( "L1_TauJet20_ETM20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet20_ETM20 ].second =
      iConfig.getParameter< double >( "L1_TauJet20_ETM20_thresh2" ) ;
   prescales_[ L1ParticleMap::kTauJet20_ETM20 ] =
      iConfig.getParameter< int >( "L1_TauJet20_ETM20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet30_ETM30 ].first =
      iConfig.getParameter< double >( "L1_TauJet30_ETM30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet30_ETM30 ].second =
      iConfig.getParameter< double >( "L1_TauJet30_ETM30_thresh2" ) ;
   prescales_[ L1ParticleMap::kTauJet30_ETM30 ] =
      iConfig.getParameter< int >( "L1_TauJet30_ETM30_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet30_ETM40 ].first =
      iConfig.getParameter< double >( "L1_TauJet30_ETM40_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kTauJet30_ETM40 ].second =
      iConfig.getParameter< double >( "L1_TauJet30_ETM40_thresh2" ) ;
   prescales_[ L1ParticleMap::kTauJet30_ETM40 ] =
      iConfig.getParameter< int >( "L1_TauJet30_ETM40_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kHTT100_ETM30 ].first =
      iConfig.getParameter< double >( "L1_HTT100_ETM30_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kHTT100_ETM30 ].second =
      iConfig.getParameter< double >( "L1_HTT100_ETM30_thresh2" ) ;
   prescales_[ L1ParticleMap::kHTT100_ETM30 ] =
      iConfig.getParameter< int >( "L1_HTT100_ETM30_prescale" ) ;

   // AAA triggers

   singleThresholds_[ L1ParticleMap::kTripleMu3 ] =
      iConfig.getParameter< double >( "L1_TripleMu3_thresh" ) ;
   prescales_[ L1ParticleMap::kTripleMu3 ] =
      iConfig.getParameter< int >( "L1_TripleMu3_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kTripleIsoEG5 ] =
      iConfig.getParameter< double >( "L1_TripleIsoEG5_thresh" ) ;
   prescales_[ L1ParticleMap::kTripleIsoEG5 ] =
      iConfig.getParameter< int >( "L1_TripleIsoEG5_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kTripleEG10 ] =
      iConfig.getParameter< double >( "L1_TripleEG10_thresh" ) ;
   prescales_[ L1ParticleMap::kTripleEG10 ] =
      iConfig.getParameter< int >( "L1_TripleEG10_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kTripleJet50 ] =
      iConfig.getParameter< double >( "L1_TripleJet50_thresh" ) ;
   prescales_[ L1ParticleMap::kTripleJet50 ] =
      iConfig.getParameter< int >( "L1_TripleJet50_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kTripleTauJet40 ] =
      iConfig.getParameter< double >( "L1_TripleTauJet40_thresh" ) ;
   prescales_[ L1ParticleMap::kTripleTauJet40 ] =
      iConfig.getParameter< int >( "L1_TripleTauJet40_prescale" ) ;

   // AAB triggers

   doubleThresholds_[ L1ParticleMap::kDoubleMu3_IsoEG5 ].first =
      iConfig.getParameter< double >( "L1_DoubleMu3_IsoEG5_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu3_IsoEG5 ].second =
      iConfig.getParameter< double >( "L1_DoubleMu3_IsoEG5_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleMu3_IsoEG5 ] =
      iConfig.getParameter< int >( "L1_DoubleMu3_IsoEG5_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu3_EG10 ].first =
      iConfig.getParameter< double >( "L1_DoubleMu3_EG10_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu3_EG10 ].second =
      iConfig.getParameter< double >( "L1_DoubleMu3_EG10_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleMu3_EG10 ] =
      iConfig.getParameter< int >( "L1_DoubleMu3_EG10_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_Mu3 ].first =
      iConfig.getParameter< double >( "L1_DoubleIsoEG5_Mu3_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_Mu3 ].second =
      iConfig.getParameter< double >( "L1_DoubleIsoEG5_Mu3_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleIsoEG5_Mu3 ] =
      iConfig.getParameter< int >( "L1_DoubleIsoEG5_Mu3_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG10_Mu3 ].first =
      iConfig.getParameter< double >( "L1_DoubleEG10_Mu3_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG10_Mu3 ].second =
      iConfig.getParameter< double >( "L1_DoubleEG10_Mu3_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleEG10_Mu3 ] =
      iConfig.getParameter< int >( "L1_DoubleEG10_Mu3_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kDoubleMu3_HTT200 ].first =
      iConfig.getParameter< double >( "L1_DoubleMu3_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu3_HTT200 ].second =
      iConfig.getParameter< double >( "L1_DoubleMu3_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleMu3_HTT200 ] =
      iConfig.getParameter< int >( "L1_DoubleMu3_HTT200_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_HTT200 ].first =
      iConfig.getParameter< double >( "L1_DoubleIsoEG5_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_HTT200 ].second =
      iConfig.getParameter< double >( "L1_DoubleIsoEG5_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleIsoEG5_HTT200 ] =
      iConfig.getParameter< int >( "L1_DoubleIsoEG5_HTT200_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG10_HTT200 ].first =
      iConfig.getParameter< double >( "L1_DoubleEG10_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG10_HTT200 ].second =
      iConfig.getParameter< double >( "L1_DoubleEG10_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleEG10_HTT200 ] =
      iConfig.getParameter< int >( "L1_DoubleEG10_HTT200_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJet50_HTT200 ].first =
      iConfig.getParameter< double >( "L1_DoubleJet50_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJet50_HTT200 ].second =
      iConfig.getParameter< double >( "L1_DoubleJet50_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleJet50_HTT200 ] =
      iConfig.getParameter< int >( "L1_DoubleJet50_HTT200_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauJet40_HTT200 ].first =
      iConfig.getParameter< double >( "L1_DoubleTauJet40_HTT200_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauJet40_HTT200 ].second =
      iConfig.getParameter< double >( "L1_DoubleTauJet40_HTT200_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleTauJet40_HTT200 ] =
      iConfig.getParameter< int >( "L1_DoubleTauJet40_HTT200_prescale" ) ;

   doubleThresholds_[ L1ParticleMap::kDoubleMu3_ETM20 ].first =
      iConfig.getParameter< double >( "L1_DoubleMu3_ETM20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleMu3_ETM20 ].second =
      iConfig.getParameter< double >( "L1_DoubleMu3_ETM20_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleMu3_ETM20 ] =
      iConfig.getParameter< int >( "L1_DoubleMu3_ETM20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_ETM20 ].first =
      iConfig.getParameter< double >( "L1_DoubleIsoEG5_ETM20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleIsoEG5_ETM20 ].second =
      iConfig.getParameter< double >( "L1_DoubleIsoEG5_ETM20_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleIsoEG5_ETM20 ] =
      iConfig.getParameter< int >( "L1_DoubleIsoEG5_ETM20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG10_ETM20 ].first =
      iConfig.getParameter< double >( "L1_DoubleEG10_ETM20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleEG10_ETM20 ].second =
      iConfig.getParameter< double >( "L1_DoubleEG10_ETM20_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleEG10_ETM20 ] =
      iConfig.getParameter< int >( "L1_DoubleEG10_ETM20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJet50_ETM20 ].first =
      iConfig.getParameter< double >( "L1_DoubleJet50_ETM20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleJet50_ETM20 ].second =
      iConfig.getParameter< double >( "L1_DoubleJet50_ETM20_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleJet50_ETM20 ] =
      iConfig.getParameter< int >( "L1_DoubleJet50_ETM20_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauJet40_ETM20 ].first =
      iConfig.getParameter< double >( "L1_DoubleTauJet40_ETM20_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kDoubleTauJet40_ETM20 ].second =
      iConfig.getParameter< double >( "L1_DoubleTauJet40_ETM20_thresh2" ) ;
   prescales_[ L1ParticleMap::kDoubleTauJet40_ETM20 ] =
      iConfig.getParameter< int >( "L1_DoubleTauJet40_ETM20_prescale" ) ;

   singleThresholds_[ L1ParticleMap::kQuadJet30 ] =
      iConfig.getParameter< double >( "L1_QuadJet30_thresh" ) ;
   prescales_[ L1ParticleMap::kQuadJet30 ] =
      iConfig.getParameter< int >( "L1_QuadJet30_prescale" ) ;

   // Diffractive triggers
   doubleThresholds_[ L1ParticleMap::kExclusiveDoubleIsoEG4 ].first =
        iConfig.getParameter< double >( "L1_ExclusiveDoubleIsoEG4_thresh1" );
   doubleThresholds_[ L1ParticleMap::kExclusiveDoubleIsoEG4 ].second =
        iConfig.getParameter< double >( "L1_ExclusiveDoubleIsoEG4_thresh2" ); // for jet rejection
   prescales_[ L1ParticleMap::kExclusiveDoubleIsoEG4 ] =
        iConfig.getParameter< int >( "L1_ExclusiveDoubleIsoEG4_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kExclusiveDoubleJet60 ] =
        iConfig.getParameter< double >( "L1_ExclusiveDoubleJet60_thresh"  ); 
   prescales_[ L1ParticleMap::kExclusiveDoubleJet60 ] =
        iConfig.getParameter< int >( "L1_ExclusiveDoubleJet60_prescale" ) ;
   singleThresholds_[ L1ParticleMap::kExclusiveJet25_Gap_Jet25 ] =
        iConfig.getParameter< double >( "L1_ExclusiveJet25_Gap_Jet25_thresh" );
   prescales_[ L1ParticleMap::kExclusiveJet25_Gap_Jet25 ] =
      iConfig.getParameter< int >( "L1_ExclusiveJet25_Gap_Jet25_prescale" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet20_ForJet10 ].first =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet20_ForJet10_thresh1" ) ;
   doubleThresholds_[ L1ParticleMap::kIsoEG10_Jet20_ForJet10 ].second =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet20_ForJet10_thresh2" ) ;
   singleThresholds_[ L1ParticleMap::kIsoEG10_Jet20_ForJet10 ] =
      iConfig.getParameter< double >( "L1_IsoEG10_Jet20_ForJet10_thresh3" ) ;
   prescales_[ L1ParticleMap::kIsoEG10_Jet20_ForJet10 ] =
      iConfig.getParameter< int >( "L1_IsoEG10_Jet20_ForJet10_prescale" ) ;

   prescales_[ L1ParticleMap::kMinBias_HTT10 ] =
      iConfig.getParameter< int >( "L1_MinBias_HTT10_prescale" ) ;
   prescales_[ L1ParticleMap::kZeroBias ] =
      iConfig.getParameter< int >( "L1_ZeroBias_prescale" ) ;

//    // Print trigger table in Twiki table format.
//    std::cout << "|  *Trigger Index*  |  *Trigger Name*  |  *E<sub>T</sub> Threshold (!GeV)*  |  *Prescale*  |"
// 	     << std::endl ;

//    for( int i = 0 ; i < L1ParticleMap::kNumOfL1TriggerTypes ; ++i )
//    {
//       std::cout
// 	 << "|  "
// 	 << i
// 	 << "  |  " ;
//       if( prescales_[ i ] == 999999999 ) std::cout << "<strike>" ;
//       std::cout
// 	 << L1ParticleMap::triggerName( ( L1ParticleMap::L1TriggerType ) i ) ;
//       if( prescales_[ i ] == 999999999 ) std::cout << "</strike>" ;
//       std::cout << "  |  " ;

//       if( singleThresholds_[ i ] > 0 )
//       {
// 	 if( doubleThresholds_[ i ].first > 0 )
// 	 {
// 	    std::cout << doubleThresholds_[ i ].first << ", "
// 		      << doubleThresholds_[ i ].second << ", " ;
// 	 }

// 	 std::cout << singleThresholds_[ i ] ;
//       }
//       else if( doubleThresholds_[ i ].first > 0 )
//       {
// 	 std::cout << doubleThresholds_[ i ].first << ", "
// 		   << doubleThresholds_[ i ].second ;
//       }
//       else
//       {
// 	 std::cout << "---" ;
//       }

//       std::cout << "  |  " ;
//       if( prescales_[ i ] != 999999999 ) std::cout << prescales_[ i ] ;
//       std::cout << "  |"
// 		<< std::endl ;
//    }
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

   Handle< L1EtMissParticle > mhtHandle ;
   iEvent.getByLabel( htMissSource_, mhtHandle ) ;

   double met = metHandle->etMiss() ;
   double ht = mhtHandle->etTotal() ;
   double ett = metHandle->etTotal() ;

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

   L1JetParticleVectorRef inputForJetRefs ;
   addToVectorRefs( forJetHandle, inputForJetRefs ) ;

   L1JetParticleVectorRef inputCenJetTauJetRefs ;
   addToVectorRefs( cenJetHandle, inputCenJetTauJetRefs ) ;
   addToVectorRefs( tauJetHandle, inputCenJetTauJetRefs ) ;

   L1MuonParticleVectorRef inputMuonRefsSingle ;
   L1MuonParticleVectorRef inputMuonRefsDouble ;
   L1MuonParticleCollection::const_iterator muItr = muHandle->begin() ;
   L1MuonParticleCollection::const_iterator muEnd = muHandle->end() ;

   for( size_t i = 0 ; muItr != muEnd ; ++muItr, ++i )
   {
      if( !muItr->gmtMuonCand().empty() )
      {
	 unsigned int qual = muItr->gmtMuonCand().quality() ;

	 if( qual == 4 ||
	     qual == 5 ||
	     qual == 6 ||
	     qual == 7 )
	 {
	    inputMuonRefsSingle.push_back(
	       edm::Ref< L1MuonParticleCollection >( muHandle, i ) ) ;
	 }

	 if( qual == 3 ||
	     qual == 5 ||
	     qual == 6 ||
	     qual == 7 )
	 {
	    inputMuonRefsDouble.push_back(
	       edm::Ref< L1MuonParticleCollection >( muHandle, i ) ) ;
	 }
      }
   }

   auto_ptr< L1ParticleMapCollection > mapColl( new L1ParticleMapCollection ) ;
   bool globalDecision = false ;
   std::vector< bool > decisionWord ;

   for( int itrig = 0 ; itrig < L1ParticleMap::kNumOfL1TriggerTypes; ++itrig )
   {
      bool decision = false ;
      std::vector< L1ParticleMap::L1ObjectType > objectTypes ;
      L1EmParticleVectorRef outputEmRefsTmp ;
      L1JetParticleVectorRef outputJetRefsTmp ;
      L1MuonParticleVectorRef outputMuonRefsTmp ;
      L1EtMissParticleRefProd metRefTmp ;
      L1ParticleMap::L1IndexComboVector combosTmp ; // unfilled for single objs

      if( itrig == L1ParticleMap::kSingleMu3 ||
	  itrig == L1ParticleMap::kSingleMu5 ||
	  itrig == L1ParticleMap::kSingleMu7 ||
	  itrig == L1ParticleMap::kSingleMu10 ||
	  itrig == L1ParticleMap::kSingleMu14 ||
	  itrig == L1ParticleMap::kSingleMu20 ||
	  itrig == L1ParticleMap::kSingleMu25 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateSingleObjectTrigger( inputMuonRefsSingle,
				      singleThresholds_[ itrig ],
				      decision,
				      outputMuonRefsTmp ) ;
      }
      else if( itrig == L1ParticleMap::kSingleIsoEG5 ||
	       itrig == L1ParticleMap::kSingleIsoEG8 ||
	       itrig == L1ParticleMap::kSingleIsoEG10 ||
	       itrig == L1ParticleMap::kSingleIsoEG12 ||
	       itrig == L1ParticleMap::kSingleIsoEG15 ||
	       itrig == L1ParticleMap::kSingleIsoEG20 ||
	       itrig == L1ParticleMap::kSingleIsoEG25 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateSingleObjectTrigger( inputIsoEmRefs,
				      singleThresholds_[ itrig ],
				      decision,
				      outputEmRefsTmp ) ;
      }
      else if( itrig == L1ParticleMap::kSingleEG5 ||
	       itrig == L1ParticleMap::kSingleEG8 ||
	       itrig == L1ParticleMap::kSingleEG10 ||
	       itrig == L1ParticleMap::kSingleEG12 ||
	       itrig == L1ParticleMap::kSingleEG15 ||
	       itrig == L1ParticleMap::kSingleEG20 ||
	       itrig == L1ParticleMap::kSingleEG25 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateSingleObjectTrigger( inputRelaxedEmRefs,
				      singleThresholds_[ itrig ],
				      decision,
				      outputEmRefsTmp ) ;
      }
      else if( itrig == L1ParticleMap::kSingleJet15 ||
	       itrig == L1ParticleMap::kSingleJet20 ||
	       itrig == L1ParticleMap::kSingleJet30 ||
	       itrig == L1ParticleMap::kSingleJet50 ||
	       itrig == L1ParticleMap::kSingleJet70 ||
	       itrig == L1ParticleMap::kSingleJet100 ||
	       itrig == L1ParticleMap::kSingleJet150 ||
	       itrig == L1ParticleMap::kSingleJet200 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateSingleObjectTrigger( inputJetRefs,
				      singleThresholds_[ itrig ],
				      decision,
				      outputJetRefsTmp ) ;
      }
      else if( itrig == L1ParticleMap::kSingleTauJet10 ||
	       itrig == L1ParticleMap::kSingleTauJet20 ||
	       itrig == L1ParticleMap::kSingleTauJet30 ||
	       itrig == L1ParticleMap::kSingleTauJet35 ||
	       itrig == L1ParticleMap::kSingleTauJet40 ||
	       itrig == L1ParticleMap::kSingleTauJet60 ||
	       itrig == L1ParticleMap::kSingleTauJet80 ||
	       itrig == L1ParticleMap::kSingleTauJet100 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateSingleObjectTrigger( inputTauRefs,
				      singleThresholds_[ itrig ],
				      decision,
				      outputJetRefsTmp ) ;
      }
      else if( itrig == L1ParticleMap::kHTT100 ||
	       itrig == L1ParticleMap::kHTT200 ||
	       itrig == L1ParticleMap::kHTT250 ||
	       itrig == L1ParticleMap::kHTT300 ||
	       itrig == L1ParticleMap::kHTT400 ||
	       itrig == L1ParticleMap::kHTT500 )
      {
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= singleThresholds_[ itrig ] )
	 {
	    decision = true ;
	    metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kETM10 ||
	       itrig == L1ParticleMap::kETM15 ||
	       itrig == L1ParticleMap::kETM20 ||
	       itrig == L1ParticleMap::kETM30 ||
	       itrig == L1ParticleMap::kETM40 ||
	       itrig == L1ParticleMap::kETM50 ||
	       itrig == L1ParticleMap::kETM60 )
      {
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= singleThresholds_[ itrig ] )
	 {
	    decision = true ;
	    metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kETT60 )
      {
	 objectTypes.push_back( L1ParticleMap::kEtTotal ) ;

	 if( ett >= singleThresholds_[ itrig ] )
	 {
	    decision = true ;
	    metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleMu3 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoubleSameObjectTrigger( inputMuonRefsDouble,
					  singleThresholds_[ itrig ],
					  decision,
					  outputMuonRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEG8 ||
	       itrig == L1ParticleMap::kDoubleIsoEG10 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleSameObjectTrigger( inputIsoEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleEG5 ||
	       itrig == L1ParticleMap::kDoubleEG10 ||
	       itrig == L1ParticleMap::kDoubleEG15 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleSameObjectTrigger( inputRelaxedEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleJet70 ||
	       itrig == L1ParticleMap::kDoubleJet100 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleSameObjectTrigger( inputJetRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleTauJet20 ||
	       itrig == L1ParticleMap::kDoubleTauJet30 ||
	       itrig == L1ParticleMap::kDoubleTauJet35 ||
	       itrig == L1ParticleMap::kDoubleTauJet40 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleSameObjectTrigger( inputTauRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kMu3_IsoEG5 ||
	       itrig == L1ParticleMap::kMu5_IsoEG10 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefsSingle,
	    inputIsoEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefsTmp,
	    outputEmRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kMu3_EG12 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefsSingle,
	    inputRelaxedEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefsTmp,
	    outputEmRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kMu3_Jet15 ||
	       itrig == L1ParticleMap::kMu5_Jet15 ||
	       itrig == L1ParticleMap::kMu3_Jet70 ||
	       itrig == L1ParticleMap::kMu5_Jet20 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefsSingle,
	    inputJetRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefsTmp,
	    outputJetRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kMu5_TauJet20 ||
	       itrig == L1ParticleMap::kMu5_TauJet30 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectTrigger(
	    inputMuonRefsSingle,
	    inputTauRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefsTmp,
	    outputJetRefsTmp,
	    combosTmp ) ;
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
	    outputEmRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kIsoEG10_Jet15 ||
	       itrig == L1ParticleMap::kIsoEG10_Jet30 ||
	       itrig == L1ParticleMap::kIsoEG10_Jet20 ||
	       itrig == L1ParticleMap::kIsoEG10_Jet70 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentCaloObjectTrigger(
	    inputIsoEmRefs,
	    inputJetRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefsTmp,
	    outputJetRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kIsoEG10_TauJet20 ||
	       itrig == L1ParticleMap::kIsoEG10_TauJet30 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentCaloObjectTrigger(
	    inputIsoEmRefs,
	    inputTauRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefsTmp,
	    outputJetRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kEG10_Jet15 ||
	       itrig == L1ParticleMap::kEG12_Jet20 ||
	       itrig == L1ParticleMap::kEG12_Jet70 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentCaloObjectTrigger(
	    inputRelaxedEmRefs,
	    inputJetRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefsTmp,
	    outputJetRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kEG12_TauJet40 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentCaloObjectTrigger(
	    inputRelaxedEmRefs,
	    inputTauRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefsTmp,
	    outputJetRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kJet70_TauJet40 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateDoubleDifferentObjectSameTypeTrigger(
	    inputJetRefs,
	    inputTauRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputJetRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kMu3_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputMuonRefsSingle,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputMuonRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kIsoEG10_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputIsoEmRefs,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputEmRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kEG12_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputRelaxedEmRefs,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputEmRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kJet70_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputJetRefs,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputJetRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kTauJet40_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputTauRefs,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputJetRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kMu3_ETM30 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputMuonRefsSingle,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputMuonRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kIsoEG10_ETM30 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputIsoEmRefs,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputEmRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kEG12_ETM30 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputRelaxedEmRefs,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputEmRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kJet70_ETM40 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputJetRefs,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputJetRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kTauJet20_ETM20 ||
	       itrig == L1ParticleMap::kTauJet30_ETM30 ||
	       itrig == L1ParticleMap::kTauJet30_ETM40 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateSingleObjectTrigger( inputTauRefs,
					 doubleThresholds_[ itrig ].first,
					 decision,
					 outputJetRefsTmp ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kHTT100_ETM30 )
      {
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( ht >= doubleThresholds_[ itrig ].first &&
	     met >= doubleThresholds_[ itrig ].second )
	 {
	    decision = true ;
	    metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kTripleMu3 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateTripleSameObjectTrigger( inputMuonRefsDouble,
					  singleThresholds_[ itrig ],
					  decision,
					  outputMuonRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kTripleIsoEG5 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateTripleSameObjectTrigger( inputIsoEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kTripleEG10 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateTripleSameObjectTrigger( inputRelaxedEmRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputEmRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kTripleJet50 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateTripleSameObjectTrigger( inputJetRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kTripleTauJet40 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateTripleSameObjectTrigger( inputTauRefs,
					  singleThresholds_[ itrig ],
					  decision,
					  outputJetRefsTmp,
					  combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleMu3_IsoEG5 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoublePlusSingleObjectTrigger(
	    inputMuonRefsDouble,
	    inputIsoEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefsTmp,
	    outputEmRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleMu3_EG10 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;

	 evaluateDoublePlusSingleObjectTrigger(
	    inputMuonRefsDouble,
	    inputRelaxedEmRefs,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputMuonRefsTmp,
	    outputEmRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEG5_Mu3 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoublePlusSingleObjectTrigger(
	    inputIsoEmRefs,
	    inputMuonRefsSingle,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefsTmp,
	    outputMuonRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleEG10_Mu3 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;

	 evaluateDoublePlusSingleObjectTrigger(
	    inputRelaxedEmRefs,
	    inputMuonRefsSingle,
	    doubleThresholds_[ itrig ].first,
	    doubleThresholds_[ itrig ].second,
	    decision,
	    outputEmRefsTmp,
	    outputMuonRefsTmp,
	    combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kDoubleMu3_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputMuonRefsDouble,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputMuonRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEG5_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputIsoEmRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputEmRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleEG10_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputRelaxedEmRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputEmRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleJet50_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputJetRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputJetRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleTauJet40_HTT200 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputTauRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputJetRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleMu3_ETM20 )
      {
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kMuon ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputMuonRefsDouble,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputMuonRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleIsoEG5_ETM20 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputIsoEmRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputEmRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleEG10_ETM20 )
      {
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEM ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputRelaxedEmRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputEmRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleJet50_ETM20 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputJetRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputJetRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kDoubleTauJet40_ETM20 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kEtMiss ) ;

	 if( met >= doubleThresholds_[ itrig ].second )
	 {
	    evaluateDoubleSameObjectTrigger( inputTauRefs,
					     doubleThresholds_[ itrig ].first,
					     decision,
					     outputJetRefsTmp,
					     combosTmp,
					     true ) ;

	    if( decision )
	    {
	       metRefTmp = L1EtMissParticleRefProd( metHandle ) ;
	    }
	 }
      }
      else if( itrig == L1ParticleMap::kQuadJet30 )
      {
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;
	 objectTypes.push_back( L1ParticleMap::kJet ) ;

	 evaluateQuadSameObjectTrigger( inputJetRefs,
					singleThresholds_[ itrig ],
					decision,
					outputJetRefsTmp,
					combosTmp ) ;
      }
      else if( itrig == L1ParticleMap::kExclusiveDoubleIsoEG4 )
      {
         objectTypes.push_back( L1ParticleMap::kEM ) ;
         objectTypes.push_back( L1ParticleMap::kEM ) ;

         evaluateDoubleExclusiveIsoEG(inputIsoEmRefs,
                                      inputJetRefs,
                                      doubleThresholds_[ itrig ].first,
                                      doubleThresholds_[ itrig ].second,
                                      decision,
                                      outputEmRefsTmp,
                                      combosTmp);

      }
      else if( itrig == L1ParticleMap::kExclusiveDoubleJet60 )
      {
         objectTypes.push_back( L1ParticleMap::kJet ) ;
         objectTypes.push_back( L1ParticleMap::kJet ) ;

         if( inputJetRefs.size() == 2 )
         {
            evaluateDoubleSameObjectTrigger( inputJetRefs,
                                             singleThresholds_[ itrig ],
                                             decision,
                                             outputJetRefsTmp,
                                             combosTmp ) ;
         }
      }
      else if( itrig == L1ParticleMap::kExclusiveJet25_Gap_Jet25 )
      {
         objectTypes.push_back( L1ParticleMap::kJet ) ;
         objectTypes.push_back( L1ParticleMap::kJet ) ;

         if( inputJetRefs.size() == 2 )
         {
            evaluateJetGapJetTrigger( inputForJetRefs,  
                                      singleThresholds_[ itrig ],
                                      decision,
                                      outputJetRefsTmp,
                                      combosTmp ) ;
         }
      }
      else if( itrig == L1ParticleMap::kIsoEG10_Jet20_ForJet10  )
      {
         objectTypes.push_back( L1ParticleMap::kEM ) ;
         objectTypes.push_back( L1ParticleMap::kJet ) ;

         evaluateForwardRapidityGap(inputForJetRefs,
                                    singleThresholds_[ itrig ],
                                    decision);

         if(decision){
           decision = false;
           evaluateDoubleDifferentCaloObjectTrigger(
              inputIsoEmRefs,
	      inputCenJetTauJetRefs,
	      doubleThresholds_[ itrig ].first,
	      doubleThresholds_[ itrig ].second,
	      decision,
	      outputEmRefsTmp,
	      outputJetRefsTmp,
	      combosTmp );
         }
      }
      else if( itrig == L1ParticleMap::kMinBias_HTT10 )
      {
	 objectTypes.push_back( L1ParticleMap::kEtHad ) ;

	 if( ht >= 10. )
	 {
	    decision = true ;
	    metRefTmp = L1EtMissParticleRefProd( mhtHandle ) ;
	 }
      }
      else if( itrig == L1ParticleMap::kZeroBias )
      {
	 decision = true ;
      }

      L1EmParticleVectorRef outputEmRefs ;
      L1JetParticleVectorRef outputJetRefs ;
      L1MuonParticleVectorRef outputMuonRefs ;
      L1EtMissParticleRefProd metRef ;
      L1ParticleMap::L1IndexComboVector combos ; // unfilled for single objs

      if( decision )
      {
//	 if( prescaleCounters_[ itrig ] % prescales_[ itrig ] )

	 double rand = CLHEP::RandFlat::shoot() * ( double ) prescales_[ itrig ] ;
	 if( rand > 1. )
	 {
	    decision = false ;
	 }
	 else
	 {
	    outputEmRefs = outputEmRefsTmp ;
	    outputJetRefs = outputJetRefsTmp ;
	    outputMuonRefs = outputMuonRefsTmp ;
	    metRef = metRefTmp ;
	    combos = combosTmp ;
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

// ok if both objects are above the threshold and are in opposite hemispheres
void L1ExtraParticleMapProd::evaluateJetGapJetTrigger(
   const l1extra::L1JetParticleVectorRef& inputRefs,        // input
   const double& etThreshold,                               // input
   bool& decision,                                          // output
   l1extra::L1JetParticleVectorRef& outputRefs,             // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )     // output
{
   // Use i+1 < inputRefs.size() instead of i < inputRefs.size()-1
   // because i is unsigned, and if size() is 0, then RHS undefined.
   for( size_t i = 0 ; i+1 < inputRefs.size() ; ++i )
   {
      const l1extra::L1JetParticleRef& refi = inputRefs[ i ] ;
      if( refi.get()->et() >= etThreshold )
      {
         for( size_t j = i+1 ; j < inputRefs.size() ; ++j )
         {
            const l1extra::L1JetParticleRef& refj = inputRefs[ j ] ;
            if( ( refj.get()->et() >= etThreshold ) &&
                ( ( ( refi.get()->eta() < 0. ) && ( refj.get()->eta() > 0. ) )
		  ||
		  ( ( refi.get()->eta() > 0. ) && ( refj.get()->eta() < 0. ) )
		  )
		)
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


// veto if both forward regions see some jet with e_T > threshold
void
L1ExtraParticleMapProd::evaluateForwardRapidityGap(
   const l1extra::L1JetParticleVectorRef& inputRefs,        // input
   const double& etThreshold,                               // input
   bool& decision                                           // output
   )
{
   decision = true;

   // search for forward pair
   for( size_t k = 0 ; k+1 < inputRefs.size() ; ++k )
   {
      const l1extra::L1JetParticleRef& refk = inputRefs[ k ] ;
      double etak = refk.get()->eta();
      if( ( refk.get()->type() == l1extra::L1JetParticle::kForward ) &&
          ( refk.get()->et()   >= etThreshold ) )
      {
          for( size_t l = k+1 ; l < inputRefs.size() ; ++l )
          {
             const l1extra::L1JetParticleRef& refl = inputRefs[ l ] ;
             double etal = refl.get()->eta();
             if( (refl.get()->type()==l1extra::L1JetParticle::kForward) &&
                 (refl.get()->et()  >= etThreshold    ) &&
                 ((etak>0 && etal<0) || (etak<0 && etal>0))    )
             {
                 decision = false ;
                 return ;// no need for going further -- for a faster algorithm
             }
          }
      }
   }
}

void
L1ExtraParticleMapProd::evaluateDoubleExclusiveIsoEG(
   const l1extra::L1EmParticleVectorRef& inputRefs1,         // input
   const l1extra::L1JetParticleVectorRef& inputRefs2,        // input
   const double& etThreshold1,                                // input
   const double& etThreshold2,                                // input
   bool& decision,                                            // output
   l1extra::L1EmParticleVectorRef& outputRefs1,               // output
   l1extra::L1ParticleMap::L1IndexComboVector& combos )       // output
{
   if ( inputRefs1.size() ==2 )
      {  // 2 iso EG
         decision=true;
         if (inputRefs2.size()>0)
            {   // should veto if there are jets, with pt>thresh
               for( size_t j = 0 ; j < inputRefs2.size() ; ++j )
                  {
                     if(inputRefs2[j].get()->gctJetCand()->regionId() ==
			inputRefs1[0].get()->gctEmCand()->regionId())continue;
                     if(inputRefs2[j].get()->gctJetCand()->regionId() ==
			inputRefs1[1].get()->gctEmCand()->regionId())continue;
                     if(inputRefs2[j].get()->et( )> etThreshold2  ) {
		       decision=false; break; }
		     // break : for a faster algorithm
                  }
            }
         if(decision)
           {   // threshold evaluation for the Exclusive double isoEG
              decision = false;
              evaluateDoubleSameObjectTrigger( inputRefs1,
                                               etThreshold1,
                                               decision,
                                               outputRefs1,
                                               combos ) ;
           }
      }
}


//define this as a plug-in
//DEFINE_FWK_MODULE(L1ExtraParticleMapProd);
