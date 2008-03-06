// Last commit: $Id: FedCablingHistosUsingDb.h,v 1.6 2008/02/07 17:02:55 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_FedCablingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FedCablingHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class FedCablingHistosUsingDb : public FedCablingHistograms, public CommissioningHistosUsingDb {
  
 public:
  
  FedCablingHistosUsingDb( DQMOldReceiver*,
			   const DbParams& );
  
  FedCablingHistosUsingDb( DQMOldReceiver*,
			   SiStripConfigDb* const );

  FedCablingHistosUsingDb( DQMStore*,
			   SiStripConfigDb* const );

  virtual ~FedCablingHistosUsingDb();
  
  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::FedConnections&,
	       const SiStripConfigDb::DeviceDescriptions&, 
	       const SiStripConfigDb::DcuDetIdMap& );
  
  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions& ) {;} // override
  
};

#endif // DQM_SiStripCommissioningClients_FedCablingHistosUsingDb_H

