// Last commit: $Id: CalibrationHistosUsingDb.h,v 1.2 2008/02/21 14:08:01 delaer Exp $

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
  
  CalibrationHistosUsingDb( DQMOldReceiver*,
			    const DbParams&,
			    const sistrip::RunType& task = sistrip::CALIBRATION );
  
  CalibrationHistosUsingDb( DQMOldReceiver*,
			    SiStripConfigDb* const,
			    const sistrip::RunType& task = sistrip::CALIBRATION );
  
  CalibrationHistosUsingDb( DQMStore*,
			    SiStripConfigDb* const,
			    const sistrip::RunType& task = sistrip::CALIBRATION );

  virtual ~CalibrationHistosUsingDb();

  virtual void uploadConfigurations();
  
 private:
  
  void update( SiStripConfigDb::DeviceDescriptions& );
  
  void create( SiStripConfigDb::AnalysisDescriptions&, Analysis );
  
};

#endif // DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H

