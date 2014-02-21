#ifndef SiPixelEDAClient_H
#define SiPixelEDAClient_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;
class SiPixelWebInterface;
class SiPixelTrackerMapCreator;
class SiPixelInformationExtractor;
class SiPixelDataQuality;
class SiPixelActionExecutor;
 
class SiPixelEDAClient: public edm::EDAnalyzer{

public:

  SiPixelEDAClient(const edm::ParameterSet& ps);
  virtual ~SiPixelEDAClient();
  
  //void defaultWebPage(xgi::Input *in, 
  //xgi::Output *out); 
  //void publish(xdata::InfoSpace *){};

protected:

  void beginJob();
  void beginRun(edm::Run const& run, 
                edm::EventSetup const& eSetup);
  void analyze(edm::Event const& e, 
               edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                            edm::EventSetup const& context) ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                          edm::EventSetup const& c);
  void endRun(edm::Run const& run, 
              edm::EventSetup const& eSetup);
  void endJob();

private:

  unsigned long long m_cacheID_;
  int nLumiSecs_;
  int nEvents_;
  int nEvents_lastLS_;
  int nErrorsBarrel_lastLS_;
  int nErrorsEndcap_lastLS_;
  
  
  DQMStore* bei_;  

  SiPixelWebInterface* sipixelWebInterface_;
  SiPixelInformationExtractor* sipixelInformationExtractor_;
  SiPixelDataQuality* sipixelDataQuality_;
  SiPixelActionExecutor* sipixelActionExecutor_;
  SiPixelTrackerMapCreator* trackerMapCreator_;

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
  bool firstRun;
  bool doHitEfficiency_;
  bool isUpgrade_;
  std::string inputSource_;
  
  std::ostringstream html_out_;
  
  //define Token(-s)
  edm::EDGetTokenT<FEDRawDataCollection> inputSourceToken_;
};


#endif
