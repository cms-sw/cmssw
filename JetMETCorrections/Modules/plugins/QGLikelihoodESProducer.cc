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
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodRcd.h"

class QGLikelihoodESProducer : public edm::ESProducer {
public:
  QGLikelihoodESProducer(const edm::ParameterSet&);
  ~QGLikelihoodESProducer() override{};

  std::shared_ptr<const QGLikelihoodObject> produce(const QGLikelihoodRcd&);

private:
  const edm::ESGetToken<QGLikelihoodObject, QGLikelihoodRcd> token_;
};

QGLikelihoodESProducer::QGLikelihoodESProducer(const edm::ParameterSet& iConfig)
    : token_(setWhatProduced(this, iConfig.getParameter<std::string>("@module_label"))
                 .consumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("algo")})) {}

// Produce the data
std::shared_ptr<const QGLikelihoodObject> QGLikelihoodESProducer::produce(const QGLikelihoodRcd& iRecord) {
  return std::shared_ptr<const QGLikelihoodObject>(&iRecord.get(token_), edm::do_nothing_deleter());
}

DEFINE_FWK_EVENTSETUP_MODULE(QGLikelihoodESProducer);
