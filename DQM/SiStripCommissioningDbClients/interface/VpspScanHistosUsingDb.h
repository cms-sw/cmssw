// Last commit: $Id: VpspScanHistosUsingDb.h,v 1.3 2007/04/04 07:21:08 bainbrid Exp $

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
  
  VpspScanHistosUsingDb( MonitorUserInterface*,
			 SiStripConfigDb* const );
  
  VpspScanHistosUsingDb( DaqMonitorBEInterface*,
			 SiStripConfigDb* const );

  virtual ~VpspScanHistosUsingDb();

  virtual void uploadToConfigDb();
  
 private:

  void update( SiStripConfigDb::DeviceDescriptions& );
  
  
};

#endif // DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

