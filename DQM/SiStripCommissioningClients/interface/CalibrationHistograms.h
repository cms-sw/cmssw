#ifndef DQM_SiStripCommissioningClients_CalibrationHistograms_H
#define DQM_SiStripCommissioningClients_CalibrationHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"


class DQMStore;

class CalibrationHistograms : virtual public CommissioningHistograms {

 public:
  
  CalibrationHistograms( const edm::ParameterSet& pset,
                         DQMStore*,
                         const sistrip::RunType& task = sistrip::CALIBRATION );
  ~CalibrationHistograms() override;
  
  void histoAnalysis( bool debug ) override;

  void printAnalyses() override; // override

 protected: 

  int calchan_;
  int isha_;
  int vfs_;
  
};

#endif // DQM_SiStripCommissioningClients_CalibrationHistograms_H

