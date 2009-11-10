// Last commit: $Id: NoiseHistosUsingDb.h,v 1.3 2009/06/18 20:52:35 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H
#define DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/NoiseHistograms.h"

class NoiseHistosUsingDb : public CommissioningHistosUsingDb, public NoiseHistograms {
  
 public:

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

