// Last commit: $Id: FineDelayHistosUsingDb.h,v 1.5 2008/03/06 13:30:50 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
#define DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class FineDelayHistosUsingDb : public CommissioningHistosUsingDb, public SamplingHistograms {
  
 public:
  
  FineDelayHistosUsingDb( DQMOldReceiver*,
			 const DbParams& );
  
  FineDelayHistosUsingDb( DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  FineDelayHistosUsingDb( DQMStore*,
			 SiStripConfigDb* const );

  virtual ~FineDelayHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  bool update( SiStripConfigDb::DeviceDescriptions& );

  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 

  void computeDelays();

  std::map<unsigned int,unsigned int > delays_;
  
};

#endif // DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
