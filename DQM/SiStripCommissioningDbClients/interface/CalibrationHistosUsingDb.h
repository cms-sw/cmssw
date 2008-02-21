// Last commit: $Id: CalibrationHistosUsingDb.h,v 1.1 2008/02/20 21:01:06 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/CalibrationHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class CalibrationHistosUsingDb : public CommissioningHistosUsingDb, public CalibrationHistograms {
  
 public:
  
  CalibrationHistosUsingDb( MonitorUserInterface*,
			    const DbParams&,
			    const sistrip::RunType& task = sistrip::CALIBRATION );
  
  CalibrationHistosUsingDb( MonitorUserInterface*,
			    SiStripConfigDb* const,
			    const sistrip::RunType& task = sistrip::CALIBRATION );
  
  CalibrationHistosUsingDb( DaqMonitorBEInterface*,
			    SiStripConfigDb* const,
			    const sistrip::RunType& task = sistrip::CALIBRATION );

  virtual ~CalibrationHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions& );
  
  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis );
  
};

#endif // DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H

