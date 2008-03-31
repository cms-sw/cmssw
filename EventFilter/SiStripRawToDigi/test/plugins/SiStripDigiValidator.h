#ifndef EventFilter_SiStripRawToDigi_SiStripDigiValidator_H
#define EventFilter_SiStripRawToDigi_SiStripDigiValidator_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

/**
    @file EventFilter/SiStripRawToDigi/test/plugins/SiStripDigiValidator.h
    @class SiStripDigiValidator

    @brief Compares two digi collections. Reports an error if the collection
           sizes do not match or the first collection conatins any digi which
           does not have an identical matching digi in the other collection.
           This guarentees that the collections are identical.
*/

class SiStripDigiValidator : public edm::EDAnalyzer {
 public:
  SiStripDigiValidator(const edm::ParameterSet& config);
  ~SiStripDigiValidator();
  virtual void beginJob(const edm::EventSetup& setup);
  virtual void endJob();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
 private:
  //Input collections
  edm::InputTag collection1Tag_;
  edm::InputTag collection2Tag_;
  bool raw_;
  //used to remember if there have been errors for message in endJob
  bool errors_;

};

// template method
template <typename T>
extern bool Compare( edm::InputTag,
		     edm::InputTag,
		     const edm::Event& event,
		     const edm::EventSetup& setup );

#endif //  EventFilter_SiStripRawToDigi_SiStripDigiValidator_H
