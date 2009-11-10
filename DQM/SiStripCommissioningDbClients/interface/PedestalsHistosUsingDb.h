// Last commit: $Id: PedestalsHistosUsingDb.h,v 1.11 2009/10/09 09:27:06 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"

class PedestalsHistosUsingDb : public CommissioningHistosUsingDb, public PedestalsHistograms {
  
 public:
  
  PedestalsHistosUsingDb( const edm::ParameterSet & pset,
                          DQMStore*,
                          SiStripConfigDb* const );
  
  virtual ~PedestalsHistosUsingDb();
  
  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::FedDescriptionsRange );

  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );

  // parameters
  float highThreshold_;
  float lowThreshold_;

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistosUsingDb_H

