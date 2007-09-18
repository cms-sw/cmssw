#ifndef L1GCTJFPARAMSCONFIGURER_H_
#define L1GCTJFPARAMSCONFIGURER_H_
// -*- C++ -*-
//
// Package:    GctConfigProducers
// Class:      L1GctJfParamsConfigurer
// 
/**\class L1GctJfParamsConfigurer L1GctJfParamsConfigurer.h L1Trigger/L1GctConfigProducers/interface/L1GctJfParamsConfigurer.h

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

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

//
// class declaration
//

class L1GctJfParamsConfigurer {
   public:
      L1GctJfParamsConfigurer(const edm::ParameterSet&);
      ~L1GctJfParamsConfigurer();

      typedef boost::shared_ptr<L1GctJetFinderParams>          JfParamsReturnType;

      JfParamsReturnType produceJfParams();

   private:
      // ----------member data ---------------------------

  // PARAMETERS TO BE STORED IN THE JetFinderParameters
  /// seed thresholds and eta boundary
  unsigned m_CenJetSeed;
  unsigned m_FwdJetSeed;
  unsigned m_TauJetSeed;
  unsigned m_EtaBoundry;

};

#endif


