// Last commit: $Id: OptoScanHistosUsingDb.h,v 1.9 2008/05/06 12:38:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"

class OptoScanHistosUsingDb : public CommissioningHistosUsingDb, public OptoScanHistograms {
  
 public:
  
  OptoScanHistosUsingDb( DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  OptoScanHistosUsingDb( DQMStore*,
			 SiStripConfigDb* const );

  virtual ~OptoScanHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ); 
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

