// Last commit: $Id: PedsOnlyHistosUsingDb.h,v 1.7 2008/03/06 13:30:50 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedsOnlyHistograms.h"

class PedsOnlyHistosUsingDb : public CommissioningHistosUsingDb, public PedsOnlyHistograms {
  
 public:
  
  PedsOnlyHistosUsingDb( DQMOldReceiver*,
			  const DbParams& );

  PedsOnlyHistosUsingDb( DQMOldReceiver*,
			  SiStripConfigDb* const );
  
  PedsOnlyHistosUsingDb( DQMStore*,
			  SiStripConfigDb* const );
  
  virtual ~PedsOnlyHistosUsingDb();
 
  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis );

};

#endif // DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H

