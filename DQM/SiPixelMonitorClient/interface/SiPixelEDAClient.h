#ifndef SiPixelEDAClient_H
#define SiPixelEDAClient_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DQMStore;
class SiPixelWebInterface;
class SiPixelTrackerMapCreator;
class SiPixelInformationExtractor;
class SiPixelActionExecutor;
 
class SiPixelEDAClient: public edm::EDAnalyzer, public evf::ModuleWeb{

public:

  SiPixelEDAClient(const edm::ParameterSet& ps);
  virtual ~SiPixelEDAClient();
  
  void defaultWebPage(xgi::Input *in, 
                      xgi::Output *out); 
  void publish(xdata::InfoSpace *){};

protected:

  void beginJob(edm::EventSetup const& eSetup);
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

  DQMStore* bei_;  

  SiPixelWebInterface* sipixelWebInterface_;
  SiPixelInformationExtractor* sipixelInformationExtractor_;
  SiPixelActionExecutor* sipixelActionExecutor_;
  SiPixelTrackerMapCreator* trackerMapCreator_;

  int tkMapFrequency_;
  int summaryFrequency_;
  unsigned int staticUpdateFrequency_;

  std::ostringstream html_out_;
};


#endif
