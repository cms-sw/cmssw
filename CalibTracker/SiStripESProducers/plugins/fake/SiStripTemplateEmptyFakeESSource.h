#ifndef CalibTracker_SiStripESProducers_SiStripTemplateEmptyFakeESSource
#define CalibTracker_SiStripESProducers_SiStripTemplateEmptyFakeESSource

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

template <typename TObject, typename TRecord>
class SiStripTemplateEmptyFakeESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiStripTemplateEmptyFakeESSource(const edm::ParameterSet&);
  SiStripTemplateEmptyFakeESSource(const SiStripTemplateEmptyFakeESSource&) = delete;
  const SiStripTemplateEmptyFakeESSource& operator=(const SiStripTemplateEmptyFakeESSource&) = delete;
  ~SiStripTemplateEmptyFakeESSource() override{};

  std::unique_ptr<TObject> produce(const TRecord&);

private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue& iov,
                      edm::ValidityInterval& iValidity) override;
};

template <typename TObject, typename TRecord>
SiStripTemplateEmptyFakeESSource<TObject, TRecord>::SiStripTemplateEmptyFakeESSource(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);
  findingRecord<TRecord>();
}

template <typename TObject, typename TRecord>
std::unique_ptr<TObject> SiStripTemplateEmptyFakeESSource<TObject, TRecord>::produce(const TRecord& iRecord) {
  return std::make_unique<TObject>();
}

template <typename TObject, typename TRecord>
void SiStripTemplateEmptyFakeESSource<TObject, TRecord>::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                                        const edm::IOVSyncValue& iov,
                                                                        edm::ValidityInterval& iValidity) {
  edm::ValidityInterval infinity(iov.beginOfTime(), iov.endOfTime());
  iValidity = infinity;
}

#endif
