// Last commit: $Id: FineDelayHistosUsingDb.h,v 1.11 2009/11/10 14:49:01 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class TrackerGeometry;

class FineDelayHistosUsingDb : public CommissioningHistosUsingDb, public SamplingHistograms {
  
 public:
  
  FineDelayHistosUsingDb( const edm::ParameterSet & pset,
                          DQMStore*,
                          SiStripConfigDb* const );

  virtual ~FineDelayHistosUsingDb();

  virtual void configure( const edm::ParameterSet&, 
                          const edm::EventSetup& );

  virtual void uploadConfigurations();

 private:
  
  bool update( SiStripConfigDb::DeviceDescriptionsRange );

  void update( SiStripConfigDb::FedDescriptionsRange );

  void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ); 

  void computeDelays();

  std::map<unsigned int,float > delays_;

  const TrackerGeometry* tracker_;
  
  bool cosmic_;
};

#endif // DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
