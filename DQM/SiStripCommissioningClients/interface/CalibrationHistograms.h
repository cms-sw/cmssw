#ifndef DQM_SiStripCommissioningClients_CalibrationHistograms_H
#define DQM_SiStripCommissioningClients_CalibrationHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class DQMOldReceiver;
class DQMStore;

class CalibrationHistograms : virtual public CommissioningHistograms {

 public:
  
  CalibrationHistograms( DQMOldReceiver*, const sistrip::RunType& task = sistrip::CALIBRATION );
  CalibrationHistograms( DQMStore*,const sistrip::RunType& task = sistrip::CALIBRATION );
  virtual ~CalibrationHistograms();
  
  void histoAnalysis( bool debug );

 protected: 

  int calchan_;
  int isha_;
  int vfs_;
  
};

#endif // DQM_SiStripCommissioningClients_CalibrationHistograms_H

