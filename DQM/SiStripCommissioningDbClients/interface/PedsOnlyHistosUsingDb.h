// Last commit: $Id: PedsOnlyHistosUsingDb.h,v 1.4 2009/11/10 14:49:01 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedsOnlyHistograms.h"

class PedsOnlyHistosUsingDb : public CommissioningHistosUsingDb, public PedsOnlyHistograms {
  
 public:

  PedsOnlyHistosUsingDb( const edm::ParameterSet & pset,
                         DQMStore*,
                         SiStripConfigDb* const );
  
  virtual ~PedsOnlyHistosUsingDb();
 
  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::FedDescriptionsRange );

  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );

};

#endif // DQM_SiStripCommissioningClients_PedsOnlyHistosUsingDb_H

