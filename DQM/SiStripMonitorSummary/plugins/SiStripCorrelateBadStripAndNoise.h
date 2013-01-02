// -*- C++ -*-
//
// Package:    SiStripCorrelateBadStripAndNoise
// Class:      SiStripCorrelateBadStripAndNoise
// 
/**\class SiStripCorrelateBadStripAndNoise SiStripCorrelateBadStripAndNoise.cc DQM/SiStripMonitorSummary/plugins/SiStripCorrelateBadStripAndNoise.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Mon Aug 10 10:42:04 CEST 2009
// $Id: SiStripCorrelateBadStripAndNoise.h,v 1.2 2009/09/23 14:20:53 kaussen Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

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
class SiStripCorrelateBadStripAndNoise : public edm::EDAnalyzer {
public:
  explicit SiStripCorrelateBadStripAndNoise(const edm::ParameterSet&);
  ~SiStripCorrelateBadStripAndNoise();
  
  
private:
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& es);
  virtual void analyze(const edm::Event&, const edm::EventSetup&){};
  virtual void endJob();
  
  void DoAnalysis(const edm::EventSetup&);
  void getHistos(const uint32_t & detid, edm::ESHandle<TrackerTopology>& tTopo, std::vector<TH2F*>& histos);
  TH2F* getHisto(const long unsigned int& index);

  unsigned long long getNoiseCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripNoisesRcd>().cacheIdentifier();}
  unsigned long long getQualityCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripQualityRcd>().cacheIdentifier();}


  void iterateOnDets(edm::ESHandle<TrackerTopology>& tTopo);
  void iterateOnBadStrips(const uint32_t & detid, edm::ESHandle<TrackerTopology>& tTopo, SiStripQuality::Range& sqrange);
  void correlateWithNoise(const uint32_t & detid, edm::ESHandle<TrackerTopology>& tTopo, const uint32_t & firstStrip,  const uint32_t & range);
  float  getMeanNoise(const SiStripNoises::Range& noiseRange,const uint32_t& first, const uint32_t& range); 


     // ----------member data ---------------------------

  SiStripDetInfoFileReader * fr;
  edm::ESHandle<SiStripQuality> qualityHandle_;
  edm::ESHandle<SiStripNoises> noiseHandle_;

  TFile* file;
  std::vector<TH2F*> vTH2;

  TrackerMap *tkmap;

  
  unsigned long long cacheID_quality;
  unsigned long long cacheID_noise;
};


