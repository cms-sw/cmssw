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
    double thePixelEfficiency[20];     // Single pixel effciency
    double thePixelColEfficiency[20];  // Column effciency
    double thePixelChipEfficiency[20]; // ROC efficiency
    std::vector<double> theLadderEfficiency_BPix[20]; // Ladder efficiency
    std::vector<double> theModuleEfficiency_BPix[20]; // Module efficiency
    std::vector<double> thePUEfficiency[20]; // Instlumi dependent efficiency
    double theInnerEfficiency_FPix[20]; // Fpix inner module efficiency
    double theOuterEfficiency_FPix[20]; // Fpix outer module efficiency
  };

#endif
