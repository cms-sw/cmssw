#ifndef SiStripQualityHotStripIdentifierRoot_H
#define SiStripQualityHotStripIdentifierRoot_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

#include "CalibTracker/SiStripQuality/interface/SiStripQualityHistos.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include <vector>
#include <memory>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <fstream>
#include <string>
#include "TFile.h"

class SiStripHotStripAlgorithmFromClusterOccupancy;
class SiStripBadAPVAlgorithmFromClusterOccupancy;
class SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy;
class TrackerTopology;

class SiStripQualityHotStripIdentifierRoot : public ConditionDBWriter<SiStripBadStrip> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  explicit SiStripQualityHotStripIdentifierRoot(const edm::ParameterSet&);
  ~SiStripQualityHotStripIdentifierRoot() override;

private:
  //Will be called at the beginning of each run in the job
  void algoBeginRun(const edm::Run&, const edm::EventSetup&) override;
  //Will be called at the beginning of each luminosity block in the run
  void algoBeginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override {}
  //Will be called at the end of the job
  void algoEndJob() override;

  //Will be called at every event
  void algoAnalyze(const edm::Event&, const edm::EventSetup&) override{};

  std::unique_ptr<SiStripBadStrip> getNewObject() override;

  void bookHistos();

private:
  bool UseInputDB_;
  const edm::ParameterSet conf_;

  const TrackerGeometry* tracker_;
  const TrackerTopology* tTopo;
  const SiStripQuality* SiStripQuality_;

  DQMStore* dqmStore_;

  TFile* file0;
  std::string filename, dirpath;
  unsigned short MinClusterWidth_, MaxClusterWidth_;
  double TotNumberOfEvents;
  double MeanNumberOfCluster;
  uint32_t calibrationthreshold;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;
  edm::ESWatcher<SiStripQualityRcd> stripQualityWatcher_;

  SiStrip::QualityHistosMap ClusterPositionHistoMap;
  SiStripHotStripAlgorithmFromClusterOccupancy* theIdentifier;
  SiStripBadAPVAlgorithmFromClusterOccupancy* theIdentifier2;
  SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy* theIdentifier3;
};
#endif
