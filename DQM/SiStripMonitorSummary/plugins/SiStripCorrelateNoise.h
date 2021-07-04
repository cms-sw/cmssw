// -*- C++ -*-
//
// Package:    SiStripCorrelateNoise
// Class:      SiStripCorrelateNoise
//
/**\class SiStripCorrelateNoise SiStripCorrelateNoise.cc
 DQM/SiStripMonitorSummary/plugins/SiStripCorrelateNoise.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Mon Aug 10 10:42:04 CEST 2009
//
//

// system include files
#include <memory>

// user include files
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

//
// class decleration
//

class TrackerTopology;
class SiStripCorrelateNoise : public edm::EDAnalyzer {
public:
  explicit SiStripCorrelateNoise(const edm::ParameterSet &);
  ~SiStripCorrelateNoise() override;

private:
  void beginRun(const edm::Run &run, const edm::EventSetup &es) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override{};
  void endJob() override;

  void DoPlots();
  void DoAnalysis(const edm::EventSetup &, const SiStripNoises &, SiStripNoises &);
  void getHistos(const uint32_t &detid, const TrackerTopology *tTopo, std::vector<TH1F *> &histos);
  TH1F *getHisto(const long unsigned int &index);

  void checkGainCache(const edm::EventSetup &es);

  float getGainRatio(const uint32_t &detid, const uint16_t &apv);

  // ----------member data ---------------------------

  struct Data {
    uint32_t detid;
    std::vector<float> values;
  };

  edm::ESWatcher<SiStripNoisesRcd> noiseWatcher_;
  edm::ESWatcher<SiStripApvGainRcd> gainWatcher_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> gainToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;

  uint32_t theRun;
  std::unique_ptr<SiStripNoises> refNoise;
  std::unique_ptr<SiStripApvGain> oldGain, newGain;
  bool equalGain;

  TFile *file;
  std::vector<TH1F *> vTH1;

  TrackerMap *tkmap;
};
