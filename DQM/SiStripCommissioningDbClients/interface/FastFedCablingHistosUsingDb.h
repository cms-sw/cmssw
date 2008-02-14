// Last commit: $Id: FastFedCablingHistosUsingDb.h,v 1.3 2008/02/07 17:02:55 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/FastFedCablingHistograms.h"

class FastFedCablingHistosUsingDb : public CommissioningHistosUsingDb, public FastFedCablingHistograms  {
  
 public:
  
  FastFedCablingHistosUsingDb( MonitorUserInterface*,
			       const DbParams& );
  
  FastFedCablingHistosUsingDb( MonitorUserInterface*,
			       SiStripConfigDb* const );

  FastFedCablingHistosUsingDb( DaqMonitorBEInterface*,
			       SiStripConfigDb* const );

  virtual ~FastFedCablingHistosUsingDb();
 
  virtual void addDcuDetIds(); // override
  
  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::FedConnections&,
	       const SiStripConfigDb::DeviceDescriptions&, 
	       const SiStripConfigDb::DcuDetIdMap& );
  
  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 

  void connections( const SiStripConfigDb::DeviceDescriptions&, 
		    const SiStripConfigDb::DcuDetIdMap& );
  
};

#endif // DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

