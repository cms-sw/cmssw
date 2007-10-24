#ifndef SiStripQualityHotStripIdentifier_H
#define SiStripQualityHotStripIdentifier_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "CalibTracker/SiStripQuality/interface/SiStripQualityHistos.h"

#include <vector>

#include <ext/hash_map>

class SiStripQualityHotStripIdentifier : public ConditionDBWriter<SiStripBadStrip> {

public:

  explicit SiStripQualityHotStripIdentifier(const edm::ParameterSet&);
  ~SiStripQualityHotStripIdentifier();

private:

 //Will be called at the beginning of the job
  void algoBeginJob(const edm::EventSetup&){resetHistos();}
  //Will be called at the beginning of each run in the job
  void algoBeginRun(const edm::Run &, const edm::EventSetup &){resetHistos();}
  //Will be called at the beginning of each luminosity block in the run
  void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){resetHistos();}

  //Will be called at every event
  void algoAnalyze(const edm::Event&, const edm::EventSetup&);

  SiStripBadStrip* getNewObject();


  void bookHistos();
  void resetHistos();
  void fillHisto(uint32_t detid,float value);

private:
  const edm::ParameterSet conf_;
  edm::FileInPath fp_;
  SiStripDetInfoFileReader* reader;
  edm::InputTag Cluster_src_;
  edm::InputTag Track_src_;
  bool tracksCollection_in_EventTree;

  unsigned short MinClusterWidth_, MaxClusterWidth_;

  SiStrip::QualityHistosMap ClusterPositionHistoMap;
};
#endif
