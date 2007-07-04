// Last commit: $Id: SiStripCommissioningDbClient.h,v 1.2 2007/03/21 16:55:06 bainbrid Exp $

#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningDbClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningDbClient_H

#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningClient.h"
#include <boost/cstdint.hpp>
#include "xdata/UnsignedLong.h"
#include "xdata/Boolean.h"
#include "xdata/String.h"
#include <string>

class SiStripConfigDb;

class SiStripCommissioningDbClient : public SiStripCommissioningClient {
  
 public:
  
  XDAQ_INSTANTIATOR();
  
  SiStripCommissioningDbClient( xdaq::ApplicationStub* );
  virtual ~SiStripCommissioningDbClient();
  
  /** */
  virtual void uploadToConfigDb();

 protected:
  
  /** */
  virtual void createHistograms( const sistrip::RunType& run_type ) const;
  
  // Extract db connections params
  xdata::Boolean usingDb_;
  xdata::String confdb_;
  xdata::String partition_;
  xdata::UnsignedLong major_;
  xdata::UnsignedLong minor_;

};

#endif // DQM_SiStripCommissioningClients_SiStripCommissioningDbClient_H

