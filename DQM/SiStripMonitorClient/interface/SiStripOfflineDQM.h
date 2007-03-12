#ifndef SiStripMonitorCluster_SiStripOfflineDQM_h
#define SiStripMonitorCluster_SiStripOfflineDQM_h
// -*- C++ -*-
//
// Package:     SiStripMonitorCluster
// Class  :     SiStripOfflineDQM
// 
/**\class SiStripOfflineDQM SiStripOfflineDQM.h DQM/SiStripMonitorCluster/interface/SiStripOfflineDQM.h

 Description: 
   Offline version of Online DQM - perform the same functionality but in
   offline mode.

 Usage:
    <usage>

*/
//
// Original Author:  Samvel Khalatyan (ksamdev at gmail dot com)
//         Created:  Wed Oct 5 16:47:14 CET 2006
//

#include <string>

// Forward classes declarations
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutorQTest.h"

class MonitorUserInterface;

class SiStripOfflineDQM: public edm::EDAnalyzer {
  public:
    explicit SiStripOfflineDQM( const edm::ParameterSet &roPARAMETER_SET);
    virtual ~SiStripOfflineDQM();

    virtual void analyze( const edm::Event	    &roEVENT, 
			                    const edm::EventSetup &roEVENT_SETUP);
    virtual void beginJob( const edm::EventSetup &roEVENT);
    virtual void endJob();

  private:
    const bool        bVERBOSE_;
    const bool        bSAVE_IN_FILE_;
    const std::string oOUT_FILE_NAME_;

    MonitorUserInterface       *poMui_;
    SiStripActionExecutorQTest  oActionExecutor_;
};

#endif
