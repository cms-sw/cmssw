
#ifndef DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H
#define DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/NoiseHistograms.h"

class NoiseHistosUsingDb : public CommissioningHistosUsingDb, public NoiseHistograms {
  
 public:

  NoiseHistosUsingDb( const edm::ParameterSet & pset,
                      DQMStore*,
                      SiStripConfigDb* const );
  
  ~NoiseHistosUsingDb() override;
 
  void uploadConfigurations() override;
  
 private:

  void update( SiStripConfigDb::FedDescriptionsRange );

  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) override;

};

#endif // DQM_SiStripCommissioningClients_NoiseHistosUsingDb_H

