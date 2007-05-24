// Last commit: $Id: FedCablingHistosUsingDb.h,v 1.3 2007/04/04 07:21:08 bainbrid Exp $

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
  
  virtual void uploadToConfigDb();
  
 private:
  
  void update( SiStripConfigDb::FedConnections&,
	       const SiStripConfigDb::DeviceDescriptions&, 
	       const SiStripConfigDb::DcuDetIdMap& );
  
  void update( SiStripConfigDb::FedDescriptions& );
  
};

#endif // DQM_SiStripCommissioningClients_FedCablingHistosUsingDb_H

