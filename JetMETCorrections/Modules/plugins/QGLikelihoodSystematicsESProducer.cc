/*
 Description: ESProducer to get the quark-gluon likelihood object "QGLikelihoodSystematicsObject"
              from record "QGLikelihoodRcd". 

 Implementation:
     Completely trivial, simply returns the QGLikelihoodSystematicsObject to the user. There is only
     one QGLikelihoodSystematicsObject object in each record. 
*/
// Original Author:  Salvatore Rappoccio
//         Created:  Thu, 13 Mar 2014 15:02:39 GMT

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodSystematicsRcd.h"

class QGLikelihoodSystematicsESProducer : public edm::ESProducer {
public:
  QGLikelihoodSystematicsESProducer(const edm::ParameterSet&);
  ~QGLikelihoodSystematicsESProducer() override{};

  std::shared_ptr<const QGLikelihoodSystematicsObject> produce(const QGLikelihoodSystematicsRcd&);

private:
  const edm::ESGetToken<QGLikelihoodSystematicsObject, QGLikelihoodSystematicsRcd> token_;
};

QGLikelihoodSystematicsESProducer::QGLikelihoodSystematicsESProducer(const edm::ParameterSet& iConfig)
    : token_(setWhatProduced(this, iConfig.getParameter<std::string>("@module_label"))
                 .consumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("algo")})) {}

// Produce the data
std::shared_ptr<const QGLikelihoodSystematicsObject> QGLikelihoodSystematicsESProducer::produce(
    const QGLikelihoodSystematicsRcd& iRecord) {
  return std::shared_ptr<const QGLikelihoodSystematicsObject>(&iRecord.get(token_), edm::do_nothing_deleter());
}

DEFINE_FWK_EVENTSETUP_MODULE(QGLikelihoodSystematicsESProducer);
