// Last commit: $Id: OptoScanHistosUsingDb.h,v 1.13 2009/11/10 14:49:01 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"

class OptoScanHistosUsingDb : public CommissioningHistosUsingDb, public OptoScanHistograms {
  
 public:
  
  OptoScanHistosUsingDb( const edm::ParameterSet & pset,
                         DQMStore*,
                         SiStripConfigDb* const );

  virtual ~OptoScanHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ); 

  // parameters
  bool skipGainUpdate_;

};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

