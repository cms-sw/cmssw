// Last commit: $Id: CalibrationHistosUsingDb.h,v 1.3 2008/03/06 13:30:50 delaer Exp $

#ifndef DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H
#define DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H

#include "DQM/SiStripCommissioningClients/interface/CalibrationHistograms.h"
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <string>
#include <map>

class TH1F;

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

  TH1F *ishaHistogram_, *vfsHistogram_; 

};

#endif // DQM_SiStripCommissioningClients_CalibrationHistosUsingDb_H

