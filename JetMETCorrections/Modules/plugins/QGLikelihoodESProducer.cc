/*
 Description: ESProducer to get the quark-gluon likelihood object "QGLikelihoodObject"
              from record "QGLikelihoodRcd". 

 Implementation:
     Completely trivial, simply returns the QGLikelihoodObject to the user. There is only
     one QGLikelihoodObject object in each record. 
*/
// Original Author:  Salvatore Rappoccio
//         Created:  Thu, 13 Mar 2014 15:02:39 GMT

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodRcd.h"

class QGLikelihoodESProducer : public edm::ESProducer {
public:
  QGLikelihoodESProducer(const edm::ParameterSet &);
  ~QGLikelihoodESProducer() override{};

  std::unique_ptr<QGLikelihoodObject> produce(const QGLikelihoodRcd &);
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &);

private:
  edm::ESGetToken<QGLikelihoodObject, QGLikelihoodRcd> token_;
};

QGLikelihoodESProducer::QGLikelihoodESProducer(const edm::ParameterSet& iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced
  std::string label = iConfig.getParameter<std::string>("@module_label");
  auto algo = iConfig.getParameter<std::string>("algo");
  setWhatProduced(this, label).setConsumes(token_, edm::ESInputTag{"", algo});
}

// The same PDF's are valid for any time
void QGLikelihoodESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                            const edm::IOVSyncValue&,
                                            edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

// Produce the data
std::unique_ptr<QGLikelihoodObject> QGLikelihoodESProducer::produce(const QGLikelihoodRcd& iRecord) {
  return std::make_unique<QGLikelihoodObject>(iRecord.get(token_));
}

DEFINE_FWK_EVENTSETUP_MODULE(QGLikelihoodESProducer);
