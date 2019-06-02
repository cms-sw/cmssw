#ifndef JetMETCorrections_Modules_QGLikelihoodESProducer_h
#define JetMETCorrections_Modules_QGLikelihoodESProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESProducts.h"
#include "CondFormats/JetMETObjects/interface/QGLikelihoodObject.h"
#include "CondFormats/DataRecord/interface/QGLikelihoodRcd.h"

class QGLikelihoodESProducer : public edm::ESProducer {
public:
  QGLikelihoodESProducer(const edm::ParameterSet &);
  ~QGLikelihoodESProducer() override{};

  std::unique_ptr<QGLikelihoodObject> produce(const QGLikelihoodRcd &);
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &);

private:
  std::string mAlgo;
};

#endif
