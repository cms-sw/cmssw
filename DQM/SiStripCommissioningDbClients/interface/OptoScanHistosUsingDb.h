// Last commit: $Id: OptoScanHistosUsingDb.h,v 1.4 2007/05/24 15:59:44 bainbrid Exp $

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
  
  OptoScanHistosUsingDb( MonitorUserInterface*,
			 SiStripConfigDb* const );
  
  OptoScanHistosUsingDb( DaqMonitorBEInterface*,
			 SiStripConfigDb* const );

  virtual ~OptoScanHistosUsingDb();
 
  virtual void addDcuDetIds();

  virtual void uploadToConfigDb();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions& );
  
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

