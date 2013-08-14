// Last commit: $Id: LatencyHistosUsingDb.h,v 1.10 2009/11/10 14:49:01 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H
#define DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class LatencyHistosUsingDb : public CommissioningHistosUsingDb, public SamplingHistograms {
  
 public:
  
  LatencyHistosUsingDb( const edm::ParameterSet & pset,
                        DQMStore*,
                        SiStripConfigDb* const );
  
  virtual ~LatencyHistosUsingDb();
  
  virtual void uploadConfigurations();

  virtual void configure( const edm::ParameterSet&, const edm::EventSetup& );
  
 private:
  
  bool update( SiStripConfigDb::DeviceDescriptionsRange, 
               SiStripConfigDb::FedDescriptionsRange );
  
  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );
  
  bool perPartition_;

};

#endif // DQM_SiStripCommissioningClients_LatencyHistosUsingDb_H

