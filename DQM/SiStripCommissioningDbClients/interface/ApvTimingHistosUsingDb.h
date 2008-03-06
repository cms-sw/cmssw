// Last commit: $Id: ApvTimingHistosUsingDb.h,v 1.8 2008/02/07 17:02:55 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"

class ApvTimingHistosUsingDb : public CommissioningHistosUsingDb, public ApvTimingHistograms  {
  
 public:
  
  ApvTimingHistosUsingDb( DQMOldReceiver*,
			  const DbParams& );

  ApvTimingHistosUsingDb( DQMOldReceiver*,
			  SiStripConfigDb* const );

  ApvTimingHistosUsingDb( DQMStore*,
			  SiStripConfigDb* const );

  virtual ~ApvTimingHistosUsingDb();
  
  virtual void uploadConfigurations();
 
  inline void uploadPllSettings( bool );
  
  inline void uploadFedSettings( bool );
  
 private:

  bool update( SiStripConfigDb::DeviceDescriptions& );
  
  void update( SiStripConfigDb::FedDescriptions& );
  
  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 
  
  bool uploadFecSettings_;

  bool uploadFedSettings_;
  
};

// ---------- Inline methods ----------

void ApvTimingHistosUsingDb::uploadPllSettings( bool upload ) { uploadFecSettings_ = upload; }
void ApvTimingHistosUsingDb::uploadFedSettings( bool upload ) { uploadFedSettings_ = upload; }

#endif // DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H

