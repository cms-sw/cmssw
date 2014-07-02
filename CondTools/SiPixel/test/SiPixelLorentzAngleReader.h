#ifndef SiPixelLorentzAngleReader_H
#define SiPixelLorentzAngleReader_H

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
#include "TROOT.h"
#include "TFile.h"
#include "TH2F.h"


//
//
// class decleration
//
  class SiPixelLorentzAngleReader : public edm::EDAnalyzer {

  public:
    explicit SiPixelLorentzAngleReader( const edm::ParameterSet& );
    ~SiPixelLorentzAngleReader();
  
    void analyze( const edm::Event&, const edm::EventSetup& );

  private:
    bool printdebug_;
    TH1F* LorentzAngleBarrel_;
    TH1F* LorentzAngleForward_;
    bool useSimRcd_;
  };

#endif
