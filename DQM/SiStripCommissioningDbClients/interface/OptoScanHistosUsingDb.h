
#ifndef DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"

class OptoScanHistosUsingDb : public CommissioningHistosUsingDb, public OptoScanHistograms {
  
 public:
  
  OptoScanHistosUsingDb( const edm::ParameterSet & pset,
                         DQMStore*,
                         SiStripConfigDb* const );

  ~OptoScanHistosUsingDb() override;

  void uploadConfigurations() override;
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) override; 

  // parameters
  bool skipGainUpdate_;
  // Perform a selective upload either for or excluding a certain set of FEDs                                                                                                                    
  bool allowSelectiveUpload_;

};

#endif // DQM_SiStripCommissioningClients_OptoScanHistosUsingDb_H

