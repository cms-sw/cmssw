// Last commit: $Id: FineDelayHistosUsingDb.h,v 1.1 2007/12/11 16:09:53 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/FineDelayHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class FineDelayHistosUsingDb : public FineDelayHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  FineDelayHistosUsingDb( MonitorUserInterface*,
			 const DbParams& );
  
  FineDelayHistosUsingDb( MonitorUserInterface*,
			 SiStripConfigDb* const );
  
  FineDelayHistosUsingDb( DaqMonitorBEInterface*,
			 SiStripConfigDb* const );

  virtual ~FineDelayHistosUsingDb();

  virtual void uploadToConfigDb();
  
 private:
  
  bool update( SiStripConfigDb::DeviceDescriptions& );

  void update( SiStripConfigDb::FedDescriptions& );
  
};

#endif // DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
