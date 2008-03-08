// Last commit: $Id: FineDelayHistosUsingDb.h,v 1.6 2008/03/06 18:16:06 delaer Exp $

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
			 const DbParams& );
  
  FineDelayHistosUsingDb( DQMOldReceiver*,
			 SiStripConfigDb* const );
  
  FineDelayHistosUsingDb( DQMStore*,
			 SiStripConfigDb* const );

  virtual ~FineDelayHistosUsingDb();

  virtual void configure(const edm::ParameterSet&, const edm::EventSetup&);

  virtual void uploadConfigurations();
  
 private:
  
  bool update( SiStripConfigDb::DeviceDescriptions& );

  void update( SiStripConfigDb::FedDescriptions& );

  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis ); 

  void computeDelays();

  std::map<unsigned int,unsigned int > delays_;

  const TrackerGeometry* tracker_;
  
};

#endif // DQM_SiStripCommissioningClients_FineDelayHistosUsingDb_H
