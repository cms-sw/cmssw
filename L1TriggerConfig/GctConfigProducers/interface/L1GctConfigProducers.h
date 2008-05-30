#ifndef L1GCTCONFIGPRODUCERS_H_
#define L1GCTCONFIGPRODUCERS_H_
// -*- C++ -*-
//
// Package:    GctConfigProducers
// Class:      L1GctConfigProducers
// 
/**\class L1GctConfigProducers L1GctConfigProducers.h L1Trigger/L1GctConfigProducers/interface/L1GctConfigProducers.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Gregory Heath
//         Created:  Thu Mar  1 15:10:47 CET 2007
// $Id: L1GctConfigProducers.h,v 1.5 2007/09/18 09:51:35 heath Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include<vector>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

class L1GctCalibFunConfigurer;
class L1GctJctSetupConfigurer;
class L1GctJfParamsConfigurer;

class L1GctJetEtCalibrationFunction;
class L1GctJetCounterSetup;
class L1GctJetFinderParams;
class L1GctChannelMask;

class L1GctJetCalibFunRcd;
class L1GctJetCounterNegativeEtaRcd;
class L1GctJetCounterPositiveEtaRcd;
class L1GctJetFinderParamsRcd;
class L1GctChannelMaskRcd;


//
// class declaration
//

class L1GctConfigProducers : public edm::ESProducer {
   public:
      L1GctConfigProducers(const edm::ParameterSet&);
      ~L1GctConfigProducers();

      typedef boost::shared_ptr<L1GctJetEtCalibrationFunction> CalibFunReturnType;
      typedef boost::shared_ptr<L1GctJetCounterSetup>          JCtSetupReturnType;
      typedef boost::shared_ptr<L1GctJetFinderParams>          JfParamsReturnType;
      typedef boost::shared_ptr<L1GctChannelMask>          ChanMaskReturnType;

      CalibFunReturnType produceCalibFun(const L1GctJetCalibFunRcd&);
      JCtSetupReturnType produceJCNegEta(const L1GctJetCounterNegativeEtaRcd&);
      JCtSetupReturnType produceJCPosEta(const L1GctJetCounterPositiveEtaRcd&);
      JfParamsReturnType produceJfParams(const L1GctJetFinderParamsRcd&);
      ChanMaskReturnType produceChanMask(const L1GctChannelMaskRcd&);



   private:
      // ----------member data ---------------------------

     L1GctCalibFunConfigurer* m_CalibFunConf;
     L1GctJctSetupConfigurer* m_JctSetupConfNegativeEta;
     L1GctJctSetupConfigurer* m_JctSetupConfPositiveEta;
     L1GctJfParamsConfigurer* m_JfParamsConf;
};

#endif


