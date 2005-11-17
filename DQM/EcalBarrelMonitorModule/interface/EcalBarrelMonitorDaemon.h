#ifndef EcalBarrelMonitorDaemon_H
#define EcalBarrelMonitorDaemon_H

/*
 * \file EcalBarrelMonitorDaemon.h
 *
 * $Date: 2005/11/14 08:52:30 $
 * $Revision: 1.11 $
 * \author G. Della Ricca
 *
*/

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace cms;
using namespace std;

class EcalBarrelMonitorDaemon{

public:

static DaqMonitorBEInterface* dbe();

protected:

/// Constructor
EcalBarrelMonitorDaemon();

/// Destructor
~EcalBarrelMonitorDaemon();

private:

static DaqMonitorBEInterface* dbe_;

};

#endif
