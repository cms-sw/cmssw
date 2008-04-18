// Last commit: $Id: FineDelayHistosUsingDb.h,v 1.4 2008/02/14 13:53:04 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/FineDelayHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class FineDelayHistosUsingDb : public CommissioningHistosUsingDb, public FineDelayHistograms {
  
 public:
  
  FineDelayHistosUsingDb( DQMOldReceiver*,
			 const DbParams& );
  
  FineDelayHistosUsingDb( DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  FineDelayHistosUsingDb( DQMStore*,
			 SiStripConfigDb* const );

  virtual ~FineDelayHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  bool update( SiStripConfigDb::DeviceDescriptions& );

  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 
  
};

#endif // DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
