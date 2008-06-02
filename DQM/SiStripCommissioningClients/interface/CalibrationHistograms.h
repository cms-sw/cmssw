#ifndef DQM_SiStripCommissioningClients_CalibrationHistograms_H
#define DQM_SiStripCommissioningClients_CalibrationHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/CalibrationSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"

class DQMOldReceiver;
class DQMStore;

class CalibrationHistograms : virtual public CommissioningHistograms {

 public:
  
  CalibrationHistograms( DQMOldReceiver*, const sistrip::RunType& task = sistrip::CALIBRATION );
  CalibrationHistograms( DQMStore*,const sistrip::RunType& task = sistrip::CALIBRATION );
  virtual ~CalibrationHistograms();
  
  typedef SummaryPlotFactory<CalibrationAnalysis*> Factory;
  typedef std::map<uint32_t,CalibrationAnalysis*> Analyses;
  
  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
                           const sistrip::Presentation&,
                           const std::string& top_level_dir,
                           const sistrip::Granularity& );
  
 protected: 
  
  Analyses data_;
  
  std::auto_ptr<Factory> factory_;

  int calchan_;
  
};

#endif // DQM_SiStripCommissioningClients_CalibrationHistograms_H

