// Last commit: $Id: VpspScanHistosUsingDb.h,v 1.8 2008/05/06 12:38:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H
#define DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"

class VpspScanHistosUsingDb : public CommissioningHistosUsingDb, public VpspScanHistograms {
  
 public:
  
  VpspScanHistosUsingDb( const edm::ParameterSet & pset,
                         DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  VpspScanHistosUsingDb( const edm::ParameterSet & pset,
                         DQMStore*,
			 SiStripConfigDb* const );

  virtual ~VpspScanHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:

  void update( SiStripConfigDb::DeviceDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );
  
};

#endif // DQM_SiStripCommissioningClients_VpspScanHistosUsingDb_H

