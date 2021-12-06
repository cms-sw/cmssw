// -*- C++ -*-
//
// Package:    SiStripPlotGain
// Class:      SiStripPlotGain
//
/**\class SiStripPlotGain SiStripPlotGain.cc
 DQM/SiStripMonitorSummary/plugins/SiStripPlotGain.cc

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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

//
// class decleration
//
class TrackerTopology;

class SiStripPlotGain : public edm::EDAnalyzer {
public:
  explicit SiStripPlotGain(const edm::ParameterSet &);
  ~SiStripPlotGain() override;

private:
  void beginRun(const edm::Run &run, const edm::EventSetup &es) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override{};
  void endJob() override;

  void DoAnalysis(const TrackerTopology &tTopo, const SiStripApvGain &);
  void getHistos(DetId detid, const TrackerTopology &tTopo, std::vector<TH1F *> &histos);
  TH1F *getHisto(const long unsigned int &index);

  // ----------member data ---------------------------

  edm::ESWatcher<SiStripApvGainRcd> gainWatcher_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> gainToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;

  TFile *file;
  std::vector<TH1F *> vTH1;

  TrackerMap *tkmap;
};
