// Last commit: $Id: ApvTimingHistosUsingDb.h,v 1.13 2009/11/15 16:42:16 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"

class ApvTimingHistosUsingDb : public CommissioningHistosUsingDb, public ApvTimingHistograms  {
  
 public:

  ApvTimingHistosUsingDb( const edm::ParameterSet & pset,
                          DQMStore*,
                          SiStripConfigDb* const );

  virtual ~ApvTimingHistosUsingDb();
  
  virtual void uploadConfigurations();

 private:

  bool update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void update( SiStripConfigDb::FedDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ); 
  
  // switch for uploading the pll thresholds
  bool skipFecUpdate_;
  // switch for uploading the frame finding thresholds
  bool skipFedUpdate_;
  
};


#endif // DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H
