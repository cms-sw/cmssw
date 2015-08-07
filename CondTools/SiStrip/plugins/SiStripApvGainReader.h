#ifndef SiStripApvGainReader_H
#define SiStripApvGainReader_H

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



class SiStripApvGainReader : public edm::EDAnalyzer {

 public:
  explicit SiStripApvGainReader( const edm::ParameterSet& );
  ~SiStripApvGainReader();
  
  void analyze( const edm::Event&, const edm::EventSetup& );

 private:
  bool printdebug_;
  std::string formatedOutput_;
  uint32_t gainType_;

};
#endif
