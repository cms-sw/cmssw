#ifndef SiStripQualityHotStripIdentifierRoot_H
#define SiStripQualityHotStripIdentifierRoot_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "CalibTracker/SiStripQuality/interface/SiStripQualityHistos.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

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

  explicit SiStripQualityHotStripIdentifierRoot(const edm::ParameterSet&);
  ~SiStripQualityHotStripIdentifierRoot() override;

private:

  //Will be called at the beginning of each run in the job
  void algoBeginRun(const edm::Run &, const edm::EventSetup &) override;
  //Will be called at the beginning of each luminosity block in the run
  void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override{  }
  //Will be called at the end of the job
  void algoEndJob() override;


  //Will be called at every event
  void algoAnalyze(const edm::Event&, const edm::EventSetup&) override{};

  std::unique_ptr<SiStripBadStrip> getNewObject() override;

  void bookHistos();

private:
  unsigned long long m_cacheID_;
  std::string dataLabel_;
  edm::ESHandle<SiStripQuality> SiStripQuality_;
  bool UseInputDB_;
  const edm::ParameterSet conf_;

  edm::ESHandle<TrackerGeometry> theTrackerGeom;
  const TrackerGeometry* tracker_;
  const TrackerTopology* tTopo;

  DQMStore* dqmStore_;

  TFile* file0;
  std::string filename, dirpath;
  unsigned short MinClusterWidth_, MaxClusterWidth_;
  double TotNumberOfEvents;
  double MeanNumberOfCluster;
  uint32_t calibrationthreshold;

  SiStrip::QualityHistosMap ClusterPositionHistoMap;
  SiStripHotStripAlgorithmFromClusterOccupancy*          theIdentifier;
  SiStripBadAPVAlgorithmFromClusterOccupancy*            theIdentifier2;
  SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy* theIdentifier3;

};
#endif
