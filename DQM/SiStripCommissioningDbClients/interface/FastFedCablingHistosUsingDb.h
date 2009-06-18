// Last commit: $Id: FastFedCablingHistosUsingDb.h,v 1.7 2008/05/06 12:38:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/FastFedCablingHistograms.h"

class FastFedCablingHistosUsingDb : public CommissioningHistosUsingDb, public FastFedCablingHistograms  {
  
 public:
  
  FastFedCablingHistosUsingDb( const edm::ParameterSet & pset,
                               DQMOldReceiver*,
			       SiStripConfigDb* const );

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

