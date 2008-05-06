// Last commit: $Id: FineDelayHistosUsingDb.h,v 1.7 2008/03/08 17:24:52 delaer Exp $

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
  
  FineDelayHistosUsingDb( DQMOldReceiver*,
			  SiStripConfigDb* const );
  
  FineDelayHistosUsingDb( DQMStore*,
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

  std::map<unsigned int,unsigned int > delays_;

  const TrackerGeometry* tracker_;
  
};

#endif // DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
