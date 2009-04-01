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
// $Id: L1GctConfigProducers.h,v 1.8 2009/02/24 15:47:23 jbrooke Exp $
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

class L1GctJfParamsConfigurer;

class L1GctJetFinderParams;
class L1GctChannelMask;

class L1GctJetFinderParamsRcd;
class L1GctChannelMaskRcd;


//
// class declaration
//

class L1GctConfigProducers : public edm::ESProducer {
   public:
      L1GctConfigProducers(const edm::ParameterSet&);
      ~L1GctConfigProducers();

      typedef boost::shared_ptr<L1GctJetFinderParams>          JfParamsReturnType;
      typedef boost::shared_ptr<L1GctChannelMask>          ChanMaskReturnType;

      JfParamsReturnType produceJfParams(const L1GctJetFinderParamsRcd&);
      ChanMaskReturnType produceChanMask(const L1GctChannelMaskRcd&);

   private:
      // ----------member data ---------------------------

     L1GctJfParamsConfigurer* m_JfParamsConf;
};

#endif


