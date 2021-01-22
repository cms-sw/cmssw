#ifndef testSiStripQualityESProducer_H
#define testSiStripQualityESProducer_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripQuality;

class testSiStripQualityESProducer : public edm::EDAnalyzer {
public:
  explicit testSiStripQualityESProducer(const edm::ParameterSet&);
  ~testSiStripQualityESProducer(){};

  void analyze(const edm::Event&, const edm::EventSetup&);

private:
  void printObject(const SiStripQuality*);
  bool printdebug_;
  unsigned long long m_cacheID_;
  bool firstIOV;
  bool twoRecordComparison_;
  std::string dataLabel_;
  std::string dataLabelTwo_;
  SiStripQuality* m_Quality_;
};
#endif
