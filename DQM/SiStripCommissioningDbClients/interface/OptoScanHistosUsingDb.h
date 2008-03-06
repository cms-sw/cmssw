// Last commit: $Id: OptoScanHistosUsingDb.h,v 1.7 2008/02/19 11:29:30 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"

class OptoScanHistosUsingDb : public CommissioningHistosUsingDb, public OptoScanHistograms {
  
 public:
  
  OptoScanHistosUsingDb( DQMOldReceiver*,
			 const DbParams& );
  
  OptoScanHistosUsingDb( DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  OptoScanHistosUsingDb( DQMStore*,
			 SiStripConfigDb* const );

  virtual ~OptoScanHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions&, const DetInfoMap& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

