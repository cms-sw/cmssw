#ifndef L1GCTJCTSETUPCONFIGURER_H_
#define L1GCTJCTSETUPCONFIGURER_H_
// -*- C++ -*-
//
// Package:    GctConfigProducers
// Class:      L1GctJctSetupConfigurer
// 
/**\class L1GctJctSetupConfigurer L1GctJctSetupConfigurer.h L1Trigger/L1GctConfigProducers/interface/L1GctJctSetupConfigurer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gregory Heath
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include<vector>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"


//
// class declaration
//

class L1GctJctSetupConfigurer {
   public:
      L1GctJctSetupConfigurer(const std::vector<edm::ParameterSet>&);
      ~L1GctJctSetupConfigurer();

      typedef boost::shared_ptr<L1GctJetCounterSetup>          JctSetupReturnType;

      JctSetupReturnType produceJctSetup();

   private:
      // ----------member data ---------------------------

  // PARAMETERS TO BE STORED IN THE JetCounterSetup
  L1GctJetCounterSetup::cutsListForWheelCard m_jetCounterCuts;

  /// member functions for setting up the jet counter configuration
  L1GctJetCounterSetup::cutsListForJetCounter addJetCounter(const edm::ParameterSet& iConfig);
  //functions for parsing strings read from config file
  L1GctJetCounterSetup::cutDescription parseDescriptor(const std::string& desc) const;
  L1GctJetCounterSetup::validCutType descToCutType (const std::string& token) const;
  unsigned                           descToCutValue(const std::string& token) const;

};

#endif


