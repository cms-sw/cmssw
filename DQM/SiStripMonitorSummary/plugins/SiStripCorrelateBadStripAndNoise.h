// -*- C++ -*-
//
// Package:    SiStripCorrelateBadStripAndNoise
// Class:      SiStripCorrelateBadStripAndNoise
//
/**\class SiStripCorrelateBadStripAndNoise SiStripCorrelateBadStripAndNoise.cc
 DQM/SiStripMonitorSummary/plugins/SiStripCorrelateBadStripAndNoise.cc

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
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
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
class TrackerGeometry;
class SiStripCorrelateBadStripAndNoise : public edm::EDAnalyzer {
public:
  explicit SiStripCorrelateBadStripAndNoise(const edm::ParameterSet &);
  ~SiStripCorrelateBadStripAndNoise() override;

private:
  void beginRun(const edm::Run &run, const edm::EventSetup &es) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override{};
  void endJob() override;

  void DoAnalysis(const edm::EventSetup &);
  void getHistos(const uint32_t &detid, const TrackerTopology &tTopo, std::vector<TH2F *> &histos);
  TH2F *getHisto(const long unsigned int &index);

  void iterateOnDets(const TrackerTopology &tTopo, const TrackerGeometry &tGeom);
  void iterateOnBadStrips(const uint32_t &detid,
                          const TrackerTopology &tTopo,
                          const TrackerGeometry &tGeom,
                          SiStripQuality::Range &sqrange);
  void correlateWithNoise(const uint32_t &detid,
                          const TrackerTopology &tTopo,
                          const uint32_t &firstStrip,
                          const uint32_t &range);
  float getMeanNoise(const SiStripNoises::Range &noiseRange, const uint32_t &first, const uint32_t &range);

  // ----------member data ---------------------------

  edm::ESWatcher<SiStripQualityRcd> qualityWatcher_;
  edm::ESWatcher<SiStripNoisesRcd> noiseWatcher_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const SiStripQuality *quality_;
  const SiStripNoises *noises_;

  TFile *file;
  std::vector<TH2F *> vTH2;

  TrackerMap *tkmap;
};
