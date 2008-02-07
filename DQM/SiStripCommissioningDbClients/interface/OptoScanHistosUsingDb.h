// Last commit: $Id: OptoScanHistosUsingDb.h,v 1.5 2007/12/19 18:18:10 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"

class OptoScanHistosUsingDb : public CommissioningHistosUsingDb, public OptoScanHistograms {
  
 public:
  
  OptoScanHistosUsingDb( MonitorUserInterface*,
			 const DbParams& );
  
  OptoScanHistosUsingDb( MonitorUserInterface*,
			 SiStripConfigDb* const );
  
  OptoScanHistosUsingDb( DaqMonitorBEInterface*,
			 SiStripConfigDb* const );

  virtual ~OptoScanHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

