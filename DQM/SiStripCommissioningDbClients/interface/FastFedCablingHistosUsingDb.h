// Last commit: $Id: FastFedCablingHistosUsingDb.h,v 1.5 2008/02/27 16:33:40 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/FastFedCablingHistograms.h"

class FastFedCablingHistosUsingDb : public CommissioningHistosUsingDb, public FastFedCablingHistograms  {
  
 public:
  
  FastFedCablingHistosUsingDb( DQMOldReceiver*,
			       const DbParams& );
  
  FastFedCablingHistosUsingDb( DQMOldReceiver*,
			       SiStripConfigDb* const );

  FastFedCablingHistosUsingDb( DQMStore*,
			       SiStripConfigDb* const );

  virtual ~FastFedCablingHistosUsingDb();
 
  virtual void addDcuDetIds(); // override
  
  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::FedConnections&,
	       const SiStripConfigDb::FedDescriptions&,
	       const SiStripConfigDb::DeviceDescriptions&, 
	       const SiStripConfigDb::DcuDetIdMap& );
  
  void update( SiStripConfigDb::FedDescriptions& );
  
  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 
  
  void connections( const SiStripConfigDb::DeviceDescriptions&, 
		    const SiStripConfigDb::DcuDetIdMap& );
  
};

#endif // DQM_SiStripCommissioningClients_FastFedCablingHistosUsingDb_H

