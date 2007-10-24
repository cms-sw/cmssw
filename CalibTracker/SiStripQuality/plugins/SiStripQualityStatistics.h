#ifndef SiStripQualityStatistics_H
#define SiStripQualityStatistics_H

// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class SiStripQualityStatistics : public edm::EDAnalyzer {

 public:
  explicit SiStripQualityStatistics( const edm::ParameterSet& );
  ~SiStripQualityStatistics(){};
  
  void analyze( const edm::Event&, const edm::EventSetup& );
    
 private:
  bool printdebug_;
  unsigned long long m_cacheID_;
  std::string dataLabel_;
};
#endif
