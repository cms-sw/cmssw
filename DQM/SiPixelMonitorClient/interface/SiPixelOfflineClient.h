#ifndef SiPixelMonitorClient_SiPixelOfflineClient_h
#define SiPixelMonitorClient_SiPixelOfflineClient_h
// -*- C++ -*-
//
// Package:	SiPixelMonitorClient
// Class  :	SiPixelOfflineClient
// 
/**\class SiPixelOfflineClient SiPixelOfflineClient.h DQM/SiPixelMonitorClient/interface/SiPixelOfflineClient.h

 Description: 
   Offline version of Online DQM - perform the same functionality but in
   offline mode.

 Usage:
    <usage>

*/
//
// Original Author:  Samvel Khalatyan (ksamdev at gmail dot com)
//	   Created:  Wed Oct 5 16:47:14 CET 2006
//

#include <string>

// Forward classes declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutorQTest.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class MonitorUserInterface;
class DaqMonitorBEInterface;
class SiPixelWebInterface;
class SiPixelTrackerMapCreator;

class SiPixelOfflineClient: public edm::EDAnalyzer, public evf::ModuleWeb{
  public:
    SiPixelOfflineClient(const edm::ParameterSet& ps_);
    virtual ~SiPixelOfflineClient();

  protected:
    void beginJob(const edm::EventSetup& es_);
    void beginRun(const edm::EventSetup& es_);
    void analyze(const edm::Event& evt_, 
		 const edm::EventSetup& es_);
    void endLuminosityBlock(const edm::LuminosityBlock& lb_,
                            const edm::EventSetup& es_);
    void defaultWebPage(xgi::Input* in, xgi::Output* out);
    void publish(xdata::InfoSpace*){};
    //void handleWebRequest(xgi::Input* in, xgi::Output* out);
    void endJob();

  private:
    int nevents_;
    //const bool verbose_;
    //const bool save_;
    const std::string outFileName_;
    edm::ParameterSet parameters_;
    
    DaqMonitorBEInterface* dbe_;
    MonitorUserInterface* mui_;
    SiPixelActionExecutorQTest ae_;
    SiPixelWebInterface* sipixelWebInterface_;
    SiPixelTrackerMapCreator* tkMapCreator_;
    
    int tkMapFreq_;
    int barrelSumFreq_;
    int endcapSumFreq_;
    int barrelGrandSumFreq_;
    int endcapGrandSumFreq_;
    int messageLimit_;
    int sourceType_;
    int saveFreq_;
    bool collFlag_;
    unsigned int run_;
    bool defPageCreated_;
    
};

#endif
