#ifndef TriggerHelper_H
#define TriggerHelper_H


// -*- C++ -*-
//
// Package:    DQM/TrackerCommon
// Class:      TriggerHelper
//
// $Id$
//
/**
  \class    TriggerHelper TriggerHelper.h "DQM/TrackerCommon/interface/TriggerHelper.h"
  \brief    Provides a code based selection for HLT path combinations in order to have no failing filters in the CMSSW path.

   [...]

  \author   Volker Adler
  \version  $Id$
*/


#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"


class TriggerHelper {

    // Data members
    // HLT configuration
    HLTConfigProvider hltConfig_;
    // Configuration parameters
    edm::InputTag hltInputTag_;
    edm::Handle< edm::TriggerResults > hltTriggerResults_;
    std::vector< std::string > hltPathNames_;
    bool andOr_;
    bool errorReply_;

  public:

    // Constructors and destructor
    TriggerHelper() { hltPathNames_.clear(); };
    ~TriggerHelper() {};

    // Public methods
    bool accept( const edm::Event & event, const edm::ParameterSet & config );

  private:

    // Private methods
    bool acceptPath( std::string hltPathName ) const;

};


#endif
