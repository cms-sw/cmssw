#ifndef SiPixelDynamicInefficiencyReader_H
#define SiPixelDynamicInefficiencyReader_H

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


//
//
// class decleration
//
  class SiPixelDynamicInefficiencyReader : public edm::EDAnalyzer {

  public:
    explicit SiPixelDynamicInefficiencyReader( const edm::ParameterSet& );
    ~SiPixelDynamicInefficiencyReader();
  
    void analyze( const edm::Event&, const edm::EventSetup& );

  private:
    bool printdebug_;
  };

#endif
