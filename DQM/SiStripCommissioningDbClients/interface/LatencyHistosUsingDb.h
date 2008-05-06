// Last commit: $Id: LatencyHistosUsingDb.h,v 1.6 2008/04/10 15:01:49 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H
#define DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class LatencyHistosUsingDb : public CommissioningHistosUsingDb, public SamplingHistograms {
  
 public:
  
  LatencyHistosUsingDb( DQMOldReceiver*,
			SiStripConfigDb* const );
  
  LatencyHistosUsingDb( DQMStore*,
			SiStripConfigDb* const );
  
  virtual ~LatencyHistosUsingDb();
  
  virtual void uploadConfigurations();
  
 private:
  
  bool update( SiStripConfigDb::DeviceDescriptionsRange, 
	       SiStripConfigDb::FedDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );
  
};

#endif // DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

