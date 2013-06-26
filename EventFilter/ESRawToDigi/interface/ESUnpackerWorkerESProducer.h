#ifndef EventFilter_ESRawToDigi_ESUnpackerWorkerESProducer_H
#define EventFilter_ESRawToDigi_ESUnpackerWorkerESProducer_H
// -*- C++ -*-
//
// Package:    ESUnpackerWorkerESProducer
// Class:      ESUnpackerWorkerESProducer
// 
/**\class ESUnpackerWorkerESProducer ESUnpackerWorkerESProducer.h EventFilter/ESUnpackerWorkerESProducer/src/ESUnpackerWorkerESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

class EcalUnpackerWorkerRecord;
class EcalUnpackerWorkerBase;
//
// class decleration
//

class ESUnpackerWorkerESProducer : public edm::ESProducer {
   public:
  ESUnpackerWorkerESProducer(const edm::ParameterSet&);
  ~ESUnpackerWorkerESProducer();

  typedef boost::shared_ptr<EcalUnpackerWorkerBase> ReturnType;
  
  ReturnType produce(const EcalUnpackerWorkerRecord & );
private:
  edm::ParameterSet conf_;
};

#endif
