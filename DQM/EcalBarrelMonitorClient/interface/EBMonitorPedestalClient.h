#ifndef EBMonitorPedestalClient_H
#define EBMonitorPedestalClient_H

/*
 * \file EBMonitorPedestalClient.h
 *
 * $Date: 2005/11/08 17:52:08 $
 * $Revision: 1.12 $
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

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class EBMonitorPedestalClient: public edm::EDAnalyzer{

friend class EcalBarrelMonitorClient;

public:

/// Constructor
EBMonitorPedestalClient(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBMonitorPedestalClient();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
virtual void endJob(void);

private:

int ievt;

};

#endif
