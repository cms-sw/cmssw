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

#include <vector>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TKey.h"
#include "TObject.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TClass.h"
class SiStripQualityHotStripIdentifierRoot : public ConditionDBWriter<SiStripBadStrip> {

public:

  explicit SiStripQualityHotStripIdentifierRoot(const edm::ParameterSet&);
  ~SiStripQualityHotStripIdentifierRoot();

private:

 //Will be called at the beginning of the job
  void algoBeginJob(const edm::EventSetup&){ }
  //Will be called at the beginning of each run in the job
  void algoBeginRun(const edm::Run &, const edm::EventSetup &){  }
  //Will be called at the beginning of each luminosity block in the run
  void algoBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &){  }
  //Will be called at the end of the job
  void algoEndJob();


  //Will be called at every event
  void algoAnalyze(const edm::Event&, const edm::EventSetup&){};

  SiStripBadStrip* getNewObject();

  void bookHistos();

private:
  const edm::ParameterSet conf_;
  edm::FileInPath fp_;
  SiStripDetInfoFileReader* reader;
  edm::InputTag Cluster_src_;
  edm::InputTag Track_src_;
  bool tracksCollection_in_EventTree;

  DQMStore* dqmStore_;

  TFile* file0;
  std::string filename, dirpath;
  unsigned short MinClusterWidth_, MaxClusterWidth_;

  SiStrip::QualityHistosMap ClusterPositionHistoMap;
};
#endif
