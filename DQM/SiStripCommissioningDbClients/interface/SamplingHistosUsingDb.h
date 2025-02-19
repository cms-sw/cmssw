// Last commit: $Id: SamplingHistosUsingDb.h,v 1.4 2009/11/10 14:49:01 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_SamplingHistosUsingDb_H
#define DQM_SiStripCommissioningClients_SamplingHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class SamplingHistosUsingDb : public CommissioningHistosUsingDb, public SamplingHistograms {
  
 public:
  
  SamplingHistosUsingDb( const edm::ParameterSet & pset,
                         DaqMonitorBEInterface*,
                         SiStripConfigDb* const );

  virtual ~SamplingHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions& );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );
  
};

#endif // DQM_SiStripCommissioningClients_SamplingHistosUsingDb_H

