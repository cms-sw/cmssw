#ifndef SiStripQualityHotStripIdentifier_H
#define SiStripQualityHotStripIdentifier_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "CalibTracker/SiStripQuality/interface/SiStripQualityHistos.h"

#include <vector>

class TrackerTopology;

class SiStripQualityHotStripIdentifier : public ConditionDBWriter<SiStripBadStrip> {

public:

  explicit SiStripQualityHotStripIdentifier(const edm::ParameterSet&);
  ~SiStripQualityHotStripIdentifier();

private:

 //Will be called at the beginning of the job
  void algoBeginJob(const edm::EventSetup&){}
  //Will be called at the beginning of each run in the job
  void algoBeginRun(const edm::Run &, const edm::EventSetup &);
  //Will be called at the beginning of each luminosity block in the run
  void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){ resetHistos(); }
  //Will be called at the end of the job
  void algoEndJob();


  //Will be called at every event
  void algoAnalyze(const edm::Event&, const edm::EventSetup&);

  SiStripBadStrip* getNewObject();


  void bookHistos();
  void resetHistos();
  void fillHisto(uint32_t detid,float value);

private:
  unsigned long long m_cacheID_;
  std::string dataLabel_;
  edm::ESHandle<SiStripQuality> SiStripQuality_;
  const edm::ParameterSet conf_;
  edm::FileInPath fp_;
  SiStripDetInfoFileReader* reader;
  edm::InputTag Cluster_src_;
  edm::InputTag Track_src_;
  bool tracksCollection_in_EventTree;
  const TrackerTopology* tTopo;

  unsigned short MinClusterWidth_, MaxClusterWidth_;

  SiStrip::QualityHistosMap ClusterPositionHistoMap;
};
#endif
