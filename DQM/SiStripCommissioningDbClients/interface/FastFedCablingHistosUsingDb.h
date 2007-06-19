// Last commit: $Id: $

#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/FastFedCablingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class FastFedCablingHistosUsingDb : public FastFedCablingHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  FastFedCablingHistosUsingDb( MonitorUserInterface*,
			       const DbParams& );
  
  FastFedCablingHistosUsingDb( MonitorUserInterface*,
			       SiStripConfigDb* const );

  FastFedCablingHistosUsingDb( DaqMonitorBEInterface*,
			       SiStripConfigDb* const );

  virtual ~FastFedCablingHistosUsingDb();
  
  virtual void uploadToConfigDb();
  
 private:
  
  void update( SiStripConfigDb::FedConnections&,
	       const SiStripConfigDb::DeviceDescriptions&, 
	       const SiStripConfigDb::DcuDetIdMap& );
  
  void update( SiStripConfigDb::FedDescriptions& );
  
};

#endif // DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

