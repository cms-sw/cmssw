// Last commit: $Id: NoiseHistosUsingDb.h,v 1.7 2008/03/06 13:30:50 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H
#define DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/NoiseHistograms.h"

class NoiseHistosUsingDb : public CommissioningHistosUsingDb, public NoiseHistograms {
  
 public:
  
  NoiseHistosUsingDb( DQMOldReceiver*,
		      const DbParams& );

  NoiseHistosUsingDb( DQMOldReceiver*,
		      SiStripConfigDb* const );
  
  NoiseHistosUsingDb( DQMStore*,
		      SiStripConfigDb* const );
  
  virtual ~NoiseHistosUsingDb();
 
  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis );

};

#endif // DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H

