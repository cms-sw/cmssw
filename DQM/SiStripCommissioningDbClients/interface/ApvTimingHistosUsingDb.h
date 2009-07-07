// Last commit: $Id: ApvTimingHistosUsingDb.h,v 1.9 2008/03/06 13:30:50 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"

class ApvTimingHistosUsingDb : public CommissioningHistosUsingDb, public ApvTimingHistograms  {
  
 public:

  ApvTimingHistosUsingDb( DQMOldReceiver*,
			  SiStripConfigDb* const );

  ApvTimingHistosUsingDb( DQMStore*,
			  SiStripConfigDb* const );

  virtual ~ApvTimingHistosUsingDb();
  
  virtual void uploadConfigurations();
 
  inline void uploadPllSettings( bool );
  
  inline void uploadFedSettings( bool );
  
 private:

  bool update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void update( SiStripConfigDb::FedDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ); 
  
  bool uploadFecSettings_;

  bool uploadFedSettings_;
  
};

// ---------- Inline methods ----------

void ApvTimingHistosUsingDb::uploadPllSettings( bool upload ) { uploadFecSettings_ = upload; }
void ApvTimingHistosUsingDb::uploadFedSettings( bool upload ) { uploadFedSettings_ = upload; }

#endif // DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H

