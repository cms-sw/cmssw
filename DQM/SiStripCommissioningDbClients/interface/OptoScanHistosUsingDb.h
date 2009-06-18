// Last commit: $Id: OptoScanHistosUsingDb.h,v 1.10 2008/07/01 12:49:35 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"

class OptoScanHistosUsingDb : public CommissioningHistosUsingDb, public OptoScanHistograms {
  
 public:
  
  OptoScanHistosUsingDb( const edm::ParameterSet & pset,
                         DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  OptoScanHistosUsingDb( const edm::ParameterSet & pset,
                         DQMStore*,
			 SiStripConfigDb* const );

  virtual ~OptoScanHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ); 
  
};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

