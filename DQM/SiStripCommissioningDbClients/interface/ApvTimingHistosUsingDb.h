
#ifndef DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"

class ApvTimingHistosUsingDb : public CommissioningHistosUsingDb, public ApvTimingHistograms  {
  
 public:

  ApvTimingHistosUsingDb( const edm::ParameterSet & pset,
                          DQMStore*,
                          SiStripConfigDb* const );

  ~ApvTimingHistosUsingDb() override;
  
  void uploadConfigurations() override;

 private:

  bool update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void update( SiStripConfigDb::FedDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) override; 
  
  // switch for uploading the pll thresholds
  bool skipFecUpdate_;
  // switch for uploading the frame finding thresholds
  bool skipFedUpdate_;
  // Perform a selective upload either for or excluding a certain set of FEDs
  bool allowSelectiveUpload_;      
};


#endif // DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H
