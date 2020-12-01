#ifndef CalibTracker_SiStripChannelGain_SiStripGainCosmicCalculator_h
#define CalibTracker_SiStripChannelGain_SiStripGainCosmicCalculator_h
// -*- C++ -*-
// Package:    SiStripApvGainCalculator
// Class:      SiStripApvGainCalculator
/**\class SiStripApvGainCalculator SiStripApvGainCalculator.cc CalibTracker/SiStripChannelGain/src/SiStripApvGainCalculator.cc
 Description: <one line class summary>
 Implementation: <Notes on implementation>
*/
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include <vector>
#include "TF1.h"
#include "TH1F.h"
#include "TObjArray.h"
#include "TFile.h"
#include "TString.h"
#include <fstream>
#include <sstream>
#include <memory>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

class SiStripGainCosmicCalculator : public ConditionDBWriter<SiStripApvGain> {
public:
  explicit SiStripGainCosmicCalculator(const edm::ParameterSet&);
  ~SiStripGainCosmicCalculator() override;

private:
  void algoAnalyze(const edm::Event&, const edm::EventSetup&) override;
  void algoBeginJob(const edm::EventSetup&) override;
  void algoEndJob() override;
  std::unique_ptr<SiStripApvGain> getNewObject() override;

private:
  std::pair<double, double> getPeakOfLandau(TH1F* inputHisto);
  double moduleWidth(const uint32_t detid);
  double moduleThickness(const uint32_t detid);

private:
  std::string TrackProducer;
  std::string TrackLabel;
  //
  TObjArray* HlistAPVPairs;
  TObjArray* HlistOtherHistos;
  uint32_t total_nr_of_events;
  double ExpectedChargeDeposition;
  std::map<uint32_t, double> thickness_map;  // map of detector id to respective thickness
  std::vector<uint32_t> SelectedDetIds;
  std::vector<uint32_t> detModulesToBeExcluded;
  unsigned int MinNrEntries;
  double MaxChi2OverNDF;
  bool outputHistogramsInRootFile;
  TString outputFileName;
  bool printdebug_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const TrackerTopology* tTopo_ = nullptr;
  const SiStripDetCabling* siStripDetCabling_ = nullptr;
  const TrackerGeometry* tkGeom_ = nullptr;
};
#endif
