// Last commit: $Id: PedestalsHistosUsingDb.h,v 1.5 2007/12/12 15:06:15 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"

class PedestalsHistosUsingDb : public CommissioningHistosUsingDb, public PedestalsHistograms {
  
 public:
  
  PedestalsHistosUsingDb( MonitorUserInterface*,
			  const DbParams& );

  PedestalsHistosUsingDb( MonitorUserInterface*,
			  SiStripConfigDb* const );
  
  PedestalsHistosUsingDb( DaqMonitorBEInterface*,
			  SiStripConfigDb* const );
  
  virtual ~PedestalsHistosUsingDb();
 
  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis );

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H

