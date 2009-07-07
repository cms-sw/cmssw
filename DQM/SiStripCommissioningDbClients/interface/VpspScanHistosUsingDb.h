// Last commit: $Id: VpspScanHistosUsingDb.h,v 1.7 2008/03/06 13:30:50 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"

class VpspScanHistosUsingDb : public CommissioningHistosUsingDb, public VpspScanHistograms {
  
 public:
  
  VpspScanHistosUsingDb( DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  VpspScanHistosUsingDb( DQMStore*,
			 SiStripConfigDb* const );

  virtual ~VpspScanHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );
  
};

#endif // DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

