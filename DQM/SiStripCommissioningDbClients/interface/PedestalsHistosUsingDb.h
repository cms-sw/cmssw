
#ifndef DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"

class PedestalsHistosUsingDb : public CommissioningHistosUsingDb, public PedestalsHistograms {
  
 public:
  
  PedestalsHistosUsingDb( const edm::ParameterSet & pset,
                          DQMStore*,
                          SiStripConfigDb* const );
  
  ~PedestalsHistosUsingDb() override;
  
  void uploadConfigurations() override;
  
 private:

  void update( SiStripConfigDb::FedDescriptionsRange );

  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) override;

  // parameters
  float highThreshold_;
  float lowThreshold_;
  bool disableBadStrips_;
  bool keepStripsDisabled_;
  // Perform a selective upload either for or excluding a certain set of FEDs                                                                                                                          
  bool allowSelectiveUpload_;

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H

