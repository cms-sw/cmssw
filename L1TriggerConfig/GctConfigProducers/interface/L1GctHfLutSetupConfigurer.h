#ifndef L1GCTHFLUTSETUPCONFIGURER_H_
#define L1GCTHFLUTSETUPCONFIGURER_H_
// -*- C++ -*-
//
// Package:    GctConfigProducers
// Class:      L1GctHfLutSetupConfigurer
// 
/**\class L1GctHfLutSetupConfigurer L1GctHfLutSetupConfigurer.h L1Trigger/L1GctConfigProducers/interface/L1GctHfLutSetupConfigurer.h

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

#include "CondFormats/L1TObjects/interface/L1GctHfLutSetup.h"


//
// class declaration
//

class L1GctHfLutSetupConfigurer {
 public:
  L1GctHfLutSetupConfigurer(const edm::ParameterSet&);
  ~L1GctHfLutSetupConfigurer();

  typedef boost::shared_ptr<L1GctHfLutSetup>          HfLutSetupReturnType;

  HfLutSetupReturnType produceHfLutSetup();

 private:
  // ----------member data ---------------------------

  // Threshold values to be stored in the HfLutSetup
  // In principle these could vary for different lut types
  // but make them all the same for now 
  std::vector<unsigned> m_thresholds;

};

#endif


