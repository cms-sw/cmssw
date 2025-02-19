#include "EventFilter/ESRawToDigi/interface/ESUnpackerWorkerESProducer.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerRecord.h"
#include "EventFilter/ESRawToDigi/interface/ESUnpackerWorker.h"

ESUnpackerWorkerESProducer::ESUnpackerWorkerESProducer(const edm::ParameterSet& iConfig)
{
  conf_ = iConfig;
  if (conf_.exists("ComponentName"))
    setWhatProduced(this,conf_.getParameter<std::string>("ComponentName"));
  else
    setWhatProduced(this);
}


ESUnpackerWorkerESProducer::~ESUnpackerWorkerESProducer(){}


ESUnpackerWorkerESProducer::ReturnType
ESUnpackerWorkerESProducer::produce(const EcalUnpackerWorkerRecord & iRecord)
{
   using namespace edm::es;

   ESUnpackerWorkerESProducer::ReturnType euw(new ESUnpackerWorker(conf_));

   //set eshandles
   euw->setHandles(iRecord);

   return euw;
}
