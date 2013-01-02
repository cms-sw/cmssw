// -*- C++ -*-
//
// Package:    SiStripPlotGain
// Class:      SiStripPlotGain
// 
/**\class SiStripPlotGain SiStripPlotGain.cc DQM/SiStripMonitorSummary/plugins/SiStripPlotGain.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Mon Aug 10 10:42:04 CEST 2009
// $Id: SiStripPlotGain.h,v 1.2 2009/09/23 14:20:53 kaussen Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"

#include "TH2F.h"
#include "TH1F.h"
#include "TFile.h"


//
// class decleration
//
class TrackerTopology;

class SiStripPlotGain : public edm::EDAnalyzer {
public:
  explicit SiStripPlotGain(const edm::ParameterSet&);
  ~SiStripPlotGain();
  
  
private:
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& es);
  virtual void analyze(const edm::Event&, const edm::EventSetup&){};
  virtual void endJob();
  
  void DoAnalysis(const edm::EventSetup& es, const SiStripApvGain&);
  void getHistos(const uint32_t & detid, edm::ESHandle<TrackerTopology>& tTopo, std::vector<TH1F*>& histos);
  TH1F* getHisto(const long unsigned int& index);

  unsigned long long getCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripApvGainRcd>().cacheIdentifier();}


     // ----------member data ---------------------------


  SiStripDetInfoFileReader * fr;

  edm::ESHandle<SiStripApvGain> Handle_;


  TFile* file;
  std::vector<TH1F*> vTH1;

  TrackerMap *tkmap;

  
  unsigned long long cacheID;
};


