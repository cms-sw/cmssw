#ifndef EcalBarrelMonitorClient_H
#define EcalBarrelMonitorClient_H

/*
 * \file EcalBarrelMonitorClient.h
 *
 * $Date: 2005/11/09 17:29:05 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBMonitorLaserClient.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBMonitorPedestalClient.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <csignal>

using namespace cms;
using namespace std;

class EcalBarrelMonitorClient: public edm::EDAnalyzer{

public:

/// Constructor
EcalBarrelMonitorClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EcalBarrelMonitorClient();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
virtual void endJob(void);

// Trap Ctrl-C
void ctrl_c_intr(int signal);

private:

int ievt_;

MonitorUserInterface* mui_;

EcalCondDBInterface* econn_;

EBMonitorLaserClient* laser_client_;
EBMonitorPedestalClient* pedestal_client_;

int exit_now;

};

DEFINE_FWK_MODULE(EcalBarrelMonitorClient)

#endif
