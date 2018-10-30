#include <memory>
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

/*
  Provide  a hook to retrieve the EcalSeverityLevelAlgo
  through the EventSetup 
 
  Appartently there is no smarter way to do it in CMSSW

  Author: Stefano Argiro
 */

class EcalSeverityLevelESProducer : public edm::ESProducer {
  
public:
  EcalSeverityLevelESProducer(const edm::ParameterSet& iConfig);
  
  typedef std::shared_ptr<EcalSeverityLevelAlgo> ReturnType;
  
  ReturnType produce(const EcalSeverityLevelAlgoRcd& iRecord);

private:

  void setupChannelStatus(const EcalChannelStatusRcd&,
                          EcalSeverityLevelAlgo*);

  using HostType = edm::ESProductHost<EcalSeverityLevelAlgo,
                                      EcalChannelStatusRcd>;

  edm::ReusableObjectHolder<HostType> holder_;
  edm::ParameterSet pset_;
};

EcalSeverityLevelESProducer::EcalSeverityLevelESProducer(const edm::ParameterSet& iConfig) :
  pset_(iConfig) {

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
}

EcalSeverityLevelESProducer::ReturnType
EcalSeverityLevelESProducer::produce(const EcalSeverityLevelAlgoRcd& iRecord){

  auto host = holder_.makeOrGet([this]() {
    return new HostType(pset_);
  });

  host->ifRecordChanges<EcalChannelStatusRcd>(iRecord,
                                              [this,h=host.get()](auto const& rec) {
    setupChannelStatus(rec, h);
  });

  return host;
}

void 
EcalSeverityLevelESProducer::setupChannelStatus(const EcalChannelStatusRcd& chs,
                                                EcalSeverityLevelAlgo* algo){
  edm::ESHandle <EcalChannelStatus> h;
  chs.get (h);
  algo->setChannelStatus(*h.product());
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalSeverityLevelESProducer);
