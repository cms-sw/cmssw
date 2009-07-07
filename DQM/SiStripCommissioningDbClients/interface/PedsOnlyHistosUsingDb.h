// Last commit: $Id: PedsOnlyHistosUsingDb.h,v 1.1 2008/03/17 17:40:54 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedsOnlyHistograms.h"

class PedsOnlyHistosUsingDb : public CommissioningHistosUsingDb, public PedsOnlyHistograms {
  
 public:

  PedsOnlyHistosUsingDb( DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  PedsOnlyHistosUsingDb( DQMStore*,
			 SiStripConfigDb* const );
  
  virtual ~PedsOnlyHistosUsingDb();
 
  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::FedDescriptionsRange );

  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );

};

#endif // DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H

