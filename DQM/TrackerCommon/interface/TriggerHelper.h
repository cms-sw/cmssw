#ifndef TriggerHelper_H
#define TriggerHelper_H


// -*- C++ -*-
//
// Package:    DQM/TrackerCommon
// Class:      TriggerHelper
//
// $Id: TriggerHelper.h,v 1.4 2010/02/16 20:22:37 vadler Exp $
//
/**
  \class    TriggerHelper TriggerHelper.h "DQM/TrackerCommon/interface/TriggerHelper.h"
  \brief    Provides a code based selection for trigger and DCS information in order to have no failing filters in the CMSSW path.

   [...]

  \author   Volker Adler
  \version  $Id: TriggerHelper.h,v 1.4 2010/02/16 20:22:37 vadler Exp $
*/


#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class TriggerHelper {

    // Data members
    // General
    bool andOr_;
    // DCS filter configuration parameters
    edm::Handle< DcsStatusCollection > dcsStatus_; // Safe here, since this is not an EDM module.
    bool errorReplyDcs_;
    // GT status bits configuration parameters
    edm::Handle< L1GlobalTriggerReadoutRecord > gtReadoutRecord_; // Safe here, since this is not an EDM module.
    bool errorReplyGt_;
    // L1 access
    L1GtUtils l1Gt_;
    // L1 filter configuration parameters
    bool errorReplyL1_;
    // HLT filter configuration parameters
    edm::InputTag hltInputTag_;
    edm::Handle< edm::TriggerResults > hltTriggerResults_; // Safe here, since this is not an EDM module.
    bool errorReplyHlt_;

  public:

    // Constructors and destructor
    TriggerHelper();
    ~TriggerHelper() {};

    // Public methods
    bool accept( const edm::Event & event, const edm::EventSetup & setup, const edm::ParameterSet & config, const HLTConfigProvider & hltConfig, const bool hltConfigInit ); // DCS, GT status, L1, HLT
    bool accept( const edm::Event & event, const edm::ParameterSet & config, const HLTConfigProvider & hltConfig, const bool hltConfigInit );                                // DCS, GT status    , HLT
    bool accept( const edm::Event & event, const edm::EventSetup & setup, const edm::ParameterSet & config );                                                                // DCS, GT status, L1
    bool accept( const edm::Event & event, const edm::ParameterSet & config );                                                                                               // DCS, GT status

  private:

    // Private methods

    // DCS
    bool acceptDcs( const edm::Event & event, const edm::ParameterSet & config );
    bool acceptDcsPartition( int dcsPartition ) const;

    // GT status bits
    bool acceptGt( const edm::Event & event, const edm::ParameterSet & config );
    bool acceptGtLogicalExpression( std::string gtLogicalExpression );

    // L1
    bool acceptL1( const edm::Event & event, const edm::EventSetup & setup, const edm::ParameterSet & config );
    bool acceptL1LogicalExpression( const edm::Event & event, std::string l1LogicalExpression );

    // HLT
    bool acceptHlt( const edm::Event & event, const edm::ParameterSet & config, const HLTConfigProvider & hltConfig, bool hltConfigInit );
    bool acceptHltLogicalExpression( std::string hltLogicalExpression, const HLTConfigProvider & hltConfig ) const;

    // Algos
    bool negate( std::string & word ) const;

};


#endif
