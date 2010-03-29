#ifndef TriggerHelper_H
#define TriggerHelper_H


// -*- C++ -*-
//
// Package:    DQM/TrackerCommon
// Class:      TriggerHelper
//
// $Id: TriggerHelper.h,v 1.1 2010/01/24 13:47:00 vadler Exp $
//
/**
  \class    TriggerHelper TriggerHelper.h "DQM/TrackerCommon/interface/TriggerHelper.h"
  \brief    Provides a code based selection for HLT path combinations in order to have no failing filters in the CMSSW path.

   [...]

  \author   Volker Adler
  \version  $Id: TriggerHelper.h,v 1.1 2010/01/24 13:47:00 vadler Exp $
*/


#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"

class TriggerHelper {

    // Data members
    // L1 access
    L1GtUtils l1Gt_;
    // L1 filter configuration parameters
    bool errorReplyL1_;
    // HLT configuration
    HLTConfigProvider hltConfig_;
    // HLT filter configuration parameters
    edm::InputTag hltInputTag_;
    edm::Handle< edm::TriggerResults > hltTriggerResults_;
    bool errorReplyHlt_;
    // DCS filter configuration parameters
    edm::Handle< DcsStatusCollection > dcsStatus_;
    bool errorReplyDcs_;

  public:

    // Constructors and destructor
    TriggerHelper();
    ~TriggerHelper() {};

    // Public methods
    bool accept( const edm::Event & event, const edm::EventSetup & setup, const edm::ParameterSet & config ); // L1, HLT and DCS combined
    bool accept( const edm::Event & event, const edm::ParameterSet & config );                                // filters for HLT and DCS only

  private:

    // Private methods

    // L1
    bool acceptL1( const edm::Event & event, const edm::EventSetup & setup, const edm::ParameterSet & config );
    bool acceptL1Algorithm( const edm::Event & event, std::string l1AlgorithmName );

    // HLT
    bool acceptHlt( const edm::Event & event, const edm::ParameterSet & config );
    bool acceptHltPath( std::string hltPathName ) const;

    // DCS
    bool acceptDcs( const edm::Event & event, const edm::ParameterSet & config );
    bool acceptDcsPartition( int dcsPartition ) const;

    // Algos
    bool negate( std::string & word ) const;

};


#endif
