#ifndef EvetnFilter_EcalRawToDigi_EcalUnpackerWorkerESProducer_H
#define EvetnFilter_EcalRawToDigi_EcalUnpackerWorkerESProducer_H
// -*- C++ -*-
//
// Package:    EcalUnpackerWorkerESProducer
// Class:      EcalUnpackerWorkerESProducer
// 
/**\class EcalUnpackerWorkerESProducer EcalUnpackerWorkerESProducer.h EventFilter/EcalUnpackerWorkerESProducer/src/EcalUnpackerWorkerESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Sat Oct  6 04:57:46 CEST 2007
// $Id: EcalUnpackerWorkerESProducer.h,v 1.2 2009/04/28 12:26:19 vlimant Exp $
//
//

#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorker.h"

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"




//
// class decleration
//

class EcalUnpackerWorkerESProducer : public edm::ESProducer {
   public:
      EcalUnpackerWorkerESProducer(const edm::ParameterSet&);
      ~EcalUnpackerWorkerESProducer();

  typedef boost::shared_ptr<EcalUnpackerWorkerBase> ReturnType;
  
  ReturnType produce(const EcalUnpackerWorkerRecord & );
private:
  edm::ParameterSet conf_;
};

#endif
