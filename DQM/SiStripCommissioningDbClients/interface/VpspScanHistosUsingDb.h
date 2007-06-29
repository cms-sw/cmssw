// Last commit: $Id: VpspScanHistosUsingDb.h,v 1.2 2007/03/21 16:55:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <map>

class VpspScanHistosUsingDb : public VpspScanHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  VpspScanHistosUsingDb( MonitorUserInterface*,
			 const DbParams& );

  VpspScanHistosUsingDb( DaqMonitorBEInterface*,
			 SiStripConfigDb* const );

  virtual ~VpspScanHistosUsingDb();

  virtual void uploadToConfigDb();
  
 private:

  void update( SiStripConfigDb::DeviceDescriptions& );
  
  
};

#endif // DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

