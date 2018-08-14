#ifndef SiStripBadStripReader_H
#define SiStripBadStripReader_H

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


class SiStripBadStripReader : public edm::EDAnalyzer {

 public:
  explicit SiStripBadStripReader( const edm::ParameterSet& );
  ~SiStripBadStripReader() override;
  
  void analyze( const edm::Event&, const edm::EventSetup& ) override;
    
 private:
  uint32_t printdebug_;
};
#endif
