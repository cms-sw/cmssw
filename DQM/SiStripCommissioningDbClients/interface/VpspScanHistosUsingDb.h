
#ifndef DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"

class VpspScanHistosUsingDb : public CommissioningHistosUsingDb, public VpspScanHistograms {
  
 public:
  
  VpspScanHistosUsingDb( const edm::ParameterSet & pset,
                         DQMStore*,
                         SiStripConfigDb* const );

  ~VpspScanHistosUsingDb() override;

  void uploadConfigurations() override;
  
 private:

  void update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) override;

  // Perform a selective upload either for or excluding a certain set of FEDs                                                                                                                      
  bool allowSelectiveUpload_;
  
};

#endif // DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

