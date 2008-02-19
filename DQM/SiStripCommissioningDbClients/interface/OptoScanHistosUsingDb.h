// Last commit: $Id: OptoScanHistosUsingDb.h,v 1.6 2008/02/07 17:02:56 bainbrid Exp $

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
  
  void update( SiStripConfigDb::DeviceDescriptions&, const DetInfoMap& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

