// Last commit: $Id: OptoScanHistosUsingDb.h,v 1.2 2007/03/21 16:55:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class OptoScanHistosUsingDb : public OptoScanHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  OptoScanHistosUsingDb( MonitorUserInterface*,
			 const DbParams& );

  OptoScanHistosUsingDb( DaqMonitorBEInterface*,
			 SiStripConfigDb* const );

  virtual ~OptoScanHistosUsingDb();

  virtual void uploadToConfigDb();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions& );
  
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

