#ifndef SiPixelEDAClient_H
#define SiPixelEDAClient_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class MonitorUserInterface;
class DaqMonitorBEInterface;
class SiPixelWebInterface;
class SiPixelTrackerMapCreator;
 
class SiPixelEDAClient: public edm::EDAnalyzer, public evf::ModuleWeb{

public:

  SiPixelEDAClient(const edm::ParameterSet& ps);
  virtual ~SiPixelEDAClient();
  
  void analyze(const edm::Event& e, 
               const edm::EventSetup& eSetup);
  void defaultWebPage(xgi::Input *in, 
                      xgi::Output *out); 
  void publish(xdata::InfoSpace *){};
  //  void handleWebRequest(xgi::Input *in, 
  //                        xgi::Output *out); 

protected:

  void beginJob(const edm::EventSetup& eSetup);
  void endRun(edm::Run const& run, 
              edm::EventSetup const& eSetup);
  void endJob();
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                            edm::EventSetup const& context) ;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                          edm::EventSetup const& c);
  //void saveAll(int irun,int ilumi);

private:

  int nLumiBlock;

  DaqMonitorBEInterface* dbe;
  MonitorUserInterface* mui_;

  edm::ParameterSet parameters;
  SiPixelWebInterface* sipixelWebInterface_;

  int tkMapFrequency_;
  int summaryFrequency_;
  unsigned int staticUpdateFrequency_;

  SiPixelTrackerMapCreator* trackerMapCreator_;
  bool defaultPageCreated_;
};


#endif
