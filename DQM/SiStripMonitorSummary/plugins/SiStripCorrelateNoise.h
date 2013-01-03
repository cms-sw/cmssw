// -*- C++ -*-
//
// Package:    SiStripCorrelateNoise
// Class:      SiStripCorrelateNoise
// 
/**\class SiStripCorrelateNoise SiStripCorrelateNoise.cc DQM/SiStripMonitorSummary/plugins/SiStripCorrelateNoise.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Mon Aug 10 10:42:04 CEST 2009
// $Id: SiStripCorrelateNoise.h,v 1.3 2013/01/02 17:23:49 wmtan Exp $
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
class SiStripCorrelateNoise : public edm::EDAnalyzer {
public:
  explicit SiStripCorrelateNoise(const edm::ParameterSet&);
  ~SiStripCorrelateNoise();
  
  
private:
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& es);
  virtual void analyze(const edm::Event&, const edm::EventSetup&){};
  virtual void endJob();
  
  void DoPlots();
  void DoAnalysis(const edm::EventSetup&,SiStripNoises,SiStripNoises&);
  void getHistos(const uint32_t & detid, const TrackerTopology* tTopo, std::vector<TH1F*>& histos);
  TH1F* getHisto(const long unsigned int& index);

  unsigned long long getNoiseCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripNoisesRcd>().cacheIdentifier();}
  unsigned long long getGainCache(const edm::EventSetup & eSetup){ return eSetup.get<SiStripApvGainRcd>().cacheIdentifier();}
  void checkGainCache(const edm::EventSetup& es);
  float  getMeanNoise(const SiStripNoises::Range& noiseRange,const uint32_t& first, const uint32_t& range); 

  float getGainRatio(const uint32_t& detid, const uint16_t& apv);

     // ----------member data ---------------------------

  struct Data{
    uint32_t detid;
    std::vector<float> values;
  };

  SiStripDetInfoFileReader * fr;
  edm::ESHandle<SiStripApvGain> gainHandle_;
  edm::ESHandle<SiStripNoises> noiseHandle_;

  uint32_t theRun;
  SiStripNoises* refNoise;

  SiStripApvGain *oldGain, *newGain;
  bool equalGain;


  TFile* file;
  std::vector<TH1F*> vTH1;

  TrackerMap *tkmap;

  
  unsigned long long cacheID_noise;
  unsigned long long cacheID_gain;
};


