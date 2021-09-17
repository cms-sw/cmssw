#ifndef SiPixelEDAClient_H
#define SiPixelEDAClient_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Stuff for cabling map to allow end run computations
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class SiPixelWebInterface;
class SiPixelInformationExtractor;
class SiPixelDataQuality;
class SiPixelActionExecutor;

class SiPixelEDAClient : public DQMEDHarvester {
public:
  SiPixelEDAClient(const edm::ParameterSet &ps);
  ~SiPixelEDAClient() override;

protected:
  void beginRun(edm::Run const &run, edm::EventSetup const &eSetup) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &iBooker,
                             DQMStore::IGetter &iGetter,
                             edm::LuminosityBlock const &lumiSeg,
                             edm::EventSetup const &c) override;
  void dqmEndJob(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) override;

private:
  unsigned long long m_cacheID_;
  int nLumiSecs_;
  int nEvents_;
  int nEvents_lastLS_;
  int nErrorsBarrel_lastLS_;
  int nErrorsEndcap_lastLS_;

  SiPixelWebInterface *sipixelWebInterface_;
  SiPixelInformationExtractor *sipixelInformationExtractor_;
  SiPixelDataQuality *sipixelDataQuality_;
  SiPixelActionExecutor *sipixelActionExecutor_;

  int tkMapFrequency_;
  int summaryFrequency_;
  unsigned int staticUpdateFrequency_;
  bool actionOnLumiSec_;
  bool actionOnRunEnd_;
  int evtOffsetForInit_;
  std::string summaryXMLfile_;
  bool hiRes_;
  double noiseRate_;
  int noiseRateDenominator_;
  bool offlineXMLfile_;
  int nFEDs_;
  bool Tier0Flag_;
  bool firstLumi;
  bool doHitEfficiency_;
  bool isUpgrade_;
  std::string inputSource_;

  std::ostringstream html_out_;

  edm::EDGetTokenT<FEDRawDataCollection> inputSourceToken_;
  SiPixelFedCablingMap theCablingMap;

  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;
};

#endif
