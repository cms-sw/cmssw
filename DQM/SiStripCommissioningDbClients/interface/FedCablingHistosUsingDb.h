// Last commit: $Id: FedCablingHistosUsingDb.h,v 1.5 2007/12/19 18:18:10 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_FedCablingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FedCablingHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class FedCablingHistosUsingDb : public FedCablingHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  FedCablingHistosUsingDb( MonitorUserInterface*,
			   const DbParams& );
  
  FedCablingHistosUsingDb( MonitorUserInterface*,
			   SiStripConfigDb* const );

  FedCablingHistosUsingDb( DaqMonitorBEInterface*,
			   SiStripConfigDb* const );

  virtual ~FedCablingHistosUsingDb();
  
  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::FedConnections&,
	       const SiStripConfigDb::DeviceDescriptions&, 
	       const SiStripConfigDb::DcuDetIdMap& );
  
  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions& ) {;} // override
  
};

#endif // DQM_SiStripCommissioningClients_FedCablingHistosUsingDb_H

