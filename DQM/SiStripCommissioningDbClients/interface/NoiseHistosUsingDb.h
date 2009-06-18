// Last commit: $Id: NoiseHistosUsingDb.h,v 1.2 2008/05/06 12:38:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H
#define DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/NoiseHistograms.h"

class NoiseHistosUsingDb : public CommissioningHistosUsingDb, public NoiseHistograms {
  
 public:

  NoiseHistosUsingDb( const edm::ParameterSet & pset,
                      DQMOldReceiver*,
		      SiStripConfigDb* const );
  
  NoiseHistosUsingDb( const edm::ParameterSet & pset,
                      DQMStore*,
		      SiStripConfigDb* const );
  
  virtual ~NoiseHistosUsingDb();
 
  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::FedDescriptionsRange );

  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );

};

#endif // DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H

