#ifndef DQM_SiStripCommissioningClients_CalibrationHistograms_H
#define DQM_SiStripCommissioningClients_CalibrationHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/CalibrationSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CalibrationAnalysis.h"

class MonitorUserInterface;

class CalibrationHistograms : public CommissioningHistograms {

 public:
  
  CalibrationHistograms( MonitorUserInterface*,const sistrip::RunType& task = sistrip::CALIBRATION );
  virtual ~CalibrationHistograms();
  
  typedef SummaryHistogramFactory<CalibrationAnalysis> Factory;
  
  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
                           const sistrip::Presentation&,
                           const std::string& top_level_dir,
                           const sistrip::Granularity& );
  
 protected: 
  
  std::map<uint32_t,CalibrationAnalysis> data_;
  
  std::auto_ptr<Factory> factory_;
  
};

#endif // DQM_SiStripCommissioningClients_CalibrationHistograms_H

