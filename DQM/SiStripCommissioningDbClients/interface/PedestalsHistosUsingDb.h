// Last commit: $Id: PedestalsHistosUsingDb.h,v 1.4 2007/05/24 15:59:44 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <map>

class PedestalsHistosUsingDb : public PedestalsHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  PedestalsHistosUsingDb( MonitorUserInterface*,
			  const DbParams& );

  PedestalsHistosUsingDb( MonitorUserInterface*,
			  SiStripConfigDb* const );
  
  PedestalsHistosUsingDb( DaqMonitorBEInterface*,
			  SiStripConfigDb* const );
  
  virtual ~PedestalsHistosUsingDb();
 
  virtual void addDcuDetIds();
 
  virtual void uploadToConfigDb();
  
 private:

  void update( SiStripConfigDb::FedDescriptions& );

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H

