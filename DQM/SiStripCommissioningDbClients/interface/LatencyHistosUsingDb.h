// Last commit: $Id: LatencyHistosUsingDb.h,v 1.3 2008/02/14 13:53:04 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H
#define DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/LatencyHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class LatencyHistosUsingDb : public CommissioningHistosUsingDb, public LatencyHistograms {
  
 public:
  
  LatencyHistosUsingDb( DQMOldReceiver*,
			 const DbParams& );
  
  LatencyHistosUsingDb( DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  LatencyHistosUsingDb( DQMStore*,
			 SiStripConfigDb* const );

  virtual ~LatencyHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions& );
  
  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis );
  
};

#endif // DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

