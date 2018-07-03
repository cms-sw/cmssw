#ifndef DQM_SiStripCommissioningClients_DaqScopeModeHistosUsingDb_H
#define DQM_SiStripCommissioningClients_DaqScopeModeHistosUsingD_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/DaqScopeModeHistograms.h"

class DaqScopeModeHistosUsingDb : public CommissioningHistosUsingDb, public DaqScopeModeHistograms {
  
 public:
  
  DaqScopeModeHistosUsingDb( const edm::ParameterSet & pset,
			     DQMStore*,
			     SiStripConfigDb* const );
  
  ~DaqScopeModeHistosUsingDb() override;
  
  void uploadConfigurations() override;
  
 private:


  void update( SiStripConfigDb::FedDescriptionsRange ); 
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) override;

  // parameters for pedestal measurement
  float highThreshold_;
  float lowThreshold_;
  bool  disableBadStrips_;
  bool  keepStripsDisabled_;

  // selective upload
  bool  allowSelectiveUpload_;    
  // switch for uploading the pll thresholds                                                                                                                                                           
  bool skipPedestalUpdate_;
  // switch for uploading the frame finding thresholds                                                                                                                                                 
  bool skipTickUpdate_;
 
};

#endif // DQM_SiStripCommissioningClients_DaqScopeModeHistosUsingDb_H

