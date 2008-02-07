// Last commit: $Id: VpspScanHistosUsingDb.h,v 1.5 2007/12/19 18:18:10 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"

class VpspScanHistosUsingDb : public CommissioningHistosUsingDb, public VpspScanHistograms {
  
 public:
  
  VpspScanHistosUsingDb( MonitorUserInterface*,
			 const DbParams& );
  
  VpspScanHistosUsingDb( MonitorUserInterface*,
			 SiStripConfigDb* const );
  
  VpspScanHistosUsingDb( DaqMonitorBEInterface*,
			 SiStripConfigDb* const );

  virtual ~VpspScanHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::DeviceDescriptions& );
  
  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis );
  
};

#endif // DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

