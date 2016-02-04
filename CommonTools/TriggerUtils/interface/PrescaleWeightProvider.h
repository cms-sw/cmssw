#ifndef CommonTools_TriggerUtils_PrescaleWeightProvider_h
#define CommonTools_TriggerUtils_PrescaleWeightProvider_h


// -*- C++ -*-
//
// Package:    CommonTools/TriggerUtils
// Class:      PrescaleWeightProvider
//
// $Id: PrescaleWeightProvider.h,v 1.1 2010/08/23 18:44:13 avetisya Exp $
//
/**
  \class    PrescaleWeightProvider PrescaleWeightProvider.h "CommonTools/TriggerUtils/interface/PrescaleWeightProvider.h"
  \brief

   This class takes a vector of HLT paths and returns a weight based on their
   HLT and L1 prescales. The weight is equal to the lowest combined (L1*HLT) prescale
   of the selected paths


  \author   Aram Avetisyan
  \version  $Id: PrescaleWeightProvider.h,v 1.1 2010/08/23 18:44:13 avetisya Exp $
*/


#include <vector>
#include <string>

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTriggerMenuLite.h"


class PrescaleWeightProvider {

    bool                               configured_;
    bool                               init_;
    HLTConfigProvider                  hltConfig_;
    edm::Handle< L1GtTriggerMenuLite > triggerMenuLite_;

    std::vector< std::string > l1SeedPaths_;

    // configuration parameters
    unsigned                   verbosity_;           // optional (default: 0)
    edm::InputTag              triggerResults_;      // optional (default: "TriggerResults::HLT")
    edm::InputTag              l1GtTriggerMenuLite_; // optional (default: "l1GtTriggerMenuLite")
    std::vector< std::string > hltPaths_;

  public:

    PrescaleWeightProvider( const edm::ParameterSet & config ); // to be called from the ED module's c'tor
    ~PrescaleWeightProvider() {}

    void initRun( const edm::Run & run, const edm::EventSetup & setup );             // to be called from the ED module's beginRun() method
    int  prescaleWeight ( const edm::Event & event, const edm::EventSetup & setup ); // to be called from the ED module's event loop method

  private:

    void parseL1Seeds( const std::string & l1Seeds );

};


#endif
