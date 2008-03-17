// Last commit: $Id: PedestalsHistosUsingDb.h,v 1.7 2008/03/06 13:30:50 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"

class PedestalsHistosUsingDb : public CommissioningHistosUsingDb, public PedestalsHistograms {
  
 public:
  
  PedestalsHistosUsingDb( DQMOldReceiver*,
			     const DbParams& );
  
  PedestalsHistosUsingDb( DQMOldReceiver*,
			     SiStripConfigDb* const );
  
  PedestalsHistosUsingDb( DQMStore*,
			     SiStripConfigDb* const );
  
  virtual ~PedestalsHistosUsingDb();
  
  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis );

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H

