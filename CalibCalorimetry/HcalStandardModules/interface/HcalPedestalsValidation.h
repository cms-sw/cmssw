#ifndef UserCode_HcalPedestalsValidation_H
#define UserCode_HcalPedestalsValidation_H

#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTBCalo/HcalPlotter/src/HcalQLPlotAnalAlgos.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalRecHit/interface/HcalCalibRecHit.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "RecoTBCalo/HcalPlotter/src/HcalQLPlotAnalAlgos.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "TFile.h"
#include "TProfile.h"
#include "TH1.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;

class HcalPedestalsValidation : public edm::EDAnalyzer {
public:
  HcalPedestalsValidation(const edm::ParameterSet& pset);
  virtual ~HcalPedestalsValidation();
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

private:

  TH1F *hbenergy;
  TH1F *hfenergy;
  TH1F *hoenergy;
  TH1F *heenergy;
  bool firsttime;
  // The file for the histograms.
  std::string outFileName;

};


#endif


