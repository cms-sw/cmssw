#ifndef SiStripQualityHotStripIdentifier_H
#define SiStripQualityHotStripIdentifier_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "CalibTracker/SiStripQuality/interface/SiStripQualityHistos.h"

#include <vector>

class TrackerTopology;

class SiStripQualityHotStripIdentifier : public ConditionDBWriter<SiStripBadStrip> {
public:
  explicit SiStripQualityHotStripIdentifier(const edm::ParameterSet &);
  ~SiStripQualityHotStripIdentifier() override;

private:
  //Will be called at the beginning of the job
  void algoBeginJob(const edm::EventSetup &) override {}
  //Will be called at the beginning of each run in the job
  void algoBeginRun(const edm::Run &, const edm::EventSetup &) override;
  //Will be called at the beginning of each luminosity block in the run
  void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override { resetHistos(); }
  //Will be called at the end of the job
  void algoEndJob() override;

  //Will be called at every event
  void algoAnalyze(const edm::Event &, const edm::EventSetup &) override;

  std::unique_ptr<SiStripBadStrip> getNewObject() override;

  void bookHistos();
  void resetHistos();
  void fillHisto(uint32_t detid, float value);

private:
  std::string dataLabel_;
  const SiStripQuality *stripQuality_ = nullptr;
  const edm::ParameterSet conf_;
  edm::FileInPath fp_;
  edm::InputTag Cluster_src_;
  edm::InputTag Track_src_;
  bool tracksCollection_in_EventTree;
  const TrackerTopology *tTopo = nullptr;

  unsigned short MinClusterWidth_, MaxClusterWidth_;

  SiStrip::QualityHistosMap ClusterPositionHistoMap;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;
  edm::ESWatcher<SiStripQualityRcd> stripQualityWatcher_;
};
#endif
