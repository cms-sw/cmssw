#ifndef EventFilter_SiStripRawToDigi_SiStripDigiValidator_H
#define EventFilter_SiStripRawToDigi_SiStripDigiValidator_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include <sstream>

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

  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);

  void validate(const edm::DetSetVector<SiStripDigi>&, const edm::DetSetVector<SiStripDigi>&);
  void validate(const edm::DetSetVector<SiStripDigi>&, const edm::DetSetVector<SiStripRawDigi>&);
  void validate(const edm::DetSetVector<SiStripRawDigi>&, const edm::DetSetVector<SiStripDigi>&);
  void validate(const edm::DetSetVector<SiStripRawDigi>&, const edm::DetSetVector<SiStripRawDigi>&);
  
 private:

  inline const std::string& header() { return header_; }
  
  //Input collections
  edm::InputTag tag1_;
  edm::InputTag tag2_;
  bool raw1_;
  bool raw2_;
  //used to remember if there have been errors for message in endJob
  bool errors_;

  std::string header_;

};

#endif //  EventFilter_SiStripRawToDigi_SiStripDigiValidator_H
