// Last commit: $Id: ApvTimingHistosUsingDb.h,v 1.5 2007/06/19 12:30:27 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class ApvTimingHistosUsingDb : public ApvTimingHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  ApvTimingHistosUsingDb( MonitorUserInterface*,
			  const DbParams& );

  ApvTimingHistosUsingDb( MonitorUserInterface*,
			  SiStripConfigDb* const );

  ApvTimingHistosUsingDb( DaqMonitorBEInterface*,
			  SiStripConfigDb* const );

  virtual ~ApvTimingHistosUsingDb();

  virtual void uploadToConfigDb();

  inline void uploadPllSettings( bool );
  
  inline void uploadFedSettings( bool );
  
 private:

  bool update( SiStripConfigDb::DeviceDescriptions& );

  void update( SiStripConfigDb::FedDescriptions& );
  
  bool uploadFecSettings_;

  bool uploadFedSettings_;
  
};

// ---------- Inline methods ----------

void ApvTimingHistosUsingDb::uploadPllSettings( bool upload ) { uploadFecSettings_ = upload; }
void ApvTimingHistosUsingDb::uploadFedSettings( bool upload ) { uploadFedSettings_ = upload; }

#endif // DQM_SiStripCommissioningClients_ApvTimingHistosUsingDb_H

