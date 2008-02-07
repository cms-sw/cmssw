// Last commit: $Id: LatencyHistosUsingDb.h,v 1.1 2007/12/11 16:09:53 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H
#define DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/LatencyHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class LatencyHistosUsingDb : public LatencyHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  LatencyHistosUsingDb( MonitorUserInterface*,
			 const DbParams& );
  
  LatencyHistosUsingDb( MonitorUserInterface*,
			 SiStripConfigDb* const );
  
  LatencyHistosUsingDb( DaqMonitorBEInterface*,
			 SiStripConfigDb* const );

  virtual ~LatencyHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions& );
  
  void create( SiStripConfigDb::AnalysisDescriptions& );
  
};

#endif // DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

