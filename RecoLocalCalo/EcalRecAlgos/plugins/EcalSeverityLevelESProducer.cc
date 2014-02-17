#include "boost/shared_ptr.hpp"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

/*
  Provide  a hook to retrieve the EcalSeverityLevelAlgo
  through the EventSetup 
 
  Appartently there is no smarter way to do it in CMSSW

  Author: Stefano Argiro
  $Id: EcalSeverityLevelESProducer.cc,v 1.1 2011/01/16 08:29:00 argiro Exp $
 */

class EcalSeverityLevelESProducer : public edm::ESProducer {
  
public:
  EcalSeverityLevelESProducer(const edm::ParameterSet& iConfig);
  
  typedef boost::shared_ptr<EcalSeverityLevelAlgo> ReturnType;
  
  ReturnType produce(const EcalSeverityLevelAlgoRcd& iRecord);
  


private:

  void chstatusCallback(const EcalChannelStatusRcd& chs);
  
  ReturnType algo_;
};

EcalSeverityLevelESProducer::EcalSeverityLevelESProducer(const edm::ParameterSet& iConfig){
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, 
		  dependsOn (&EcalSeverityLevelESProducer::chstatusCallback));

  algo_ = ReturnType(new EcalSeverityLevelAlgo(iConfig));
}



EcalSeverityLevelESProducer::ReturnType
EcalSeverityLevelESProducer::produce(const EcalSeverityLevelAlgoRcd& iRecord){
  
  return algo_ ;
}


void 
EcalSeverityLevelESProducer::chstatusCallback(const EcalChannelStatusRcd& chs){
  edm::ESHandle <EcalChannelStatus> h;
  chs.get (h);
  algo_->setChannelStatus(*h.product());
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalSeverityLevelESProducer);


