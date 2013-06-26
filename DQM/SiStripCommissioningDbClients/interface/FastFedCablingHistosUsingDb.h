// Last commit: $Id: FastFedCablingHistosUsingDb.h,v 1.9 2009/11/10 14:49:01 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/FastFedCablingHistograms.h"

class FastFedCablingHistosUsingDb : public CommissioningHistosUsingDb, public FastFedCablingHistograms  {
  
 public:
  
  FastFedCablingHistosUsingDb( const edm::ParameterSet & pset,
                               DQMStore*,
                               SiStripConfigDb* const );

  virtual ~FastFedCablingHistosUsingDb();
 
  virtual void addDcuDetIds(); // override
  
  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::FedConnectionsV&,
               SiStripConfigDb::FedDescriptionsRange,
               SiStripConfigDb::DeviceDescriptionsRange, 
               SiStripConfigDb::DcuDetIdsRange );
  
  void update( SiStripConfigDb::FedDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ); 
  
  void connections( SiStripConfigDb::DeviceDescriptionsRange, 
                    SiStripConfigDb::DcuDetIdsRange );
  
};

#endif // DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

