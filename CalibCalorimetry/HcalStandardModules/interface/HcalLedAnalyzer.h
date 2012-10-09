#ifndef HcalLedAnalyzer_H
#define HcalLedAnalyzer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalLedAnalysis.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class HcalLedAnalyzer: public edm::EDAnalyzer{

public:

/// Constructor
HcalLedAnalyzer(const edm::ParameterSet& ps);

/// Destructor
~HcalLedAnalyzer();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob();

// EndJob
void endJob(void);

private:
 
  int m_ievt;
  int led_sample;
  //  string m_outputFileROOT;
  HcalLedAnalysis* m_ledAnal;
  HcalPedestals* m_inputPeds;

//  int m_startSample;
//  int m_endSample;

  std::string m_inputPedestals_source;
  std::string m_inputPedestals_tag;
  int m_inputPedestals_run;

  edm::InputTag hbheDigiCollectionTag_;
  edm::InputTag hoDigiCollectionTag_;
  edm::InputTag hfDigiCollectionTag_;
  edm::InputTag hcalCalibDigiCollectionTag_;
};

#endif
