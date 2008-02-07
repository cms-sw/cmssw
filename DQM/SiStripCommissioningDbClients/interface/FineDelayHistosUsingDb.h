// Last commit: $Id: FineDelayHistosUsingDb.h,v 1.2 2007/12/11 17:11:12 delaer Exp $

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

  virtual void uploadConfigurations();
  
 private:
  
  bool update( SiStripConfigDb::DeviceDescriptions& );

  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions& );
  
};

#endif // DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
