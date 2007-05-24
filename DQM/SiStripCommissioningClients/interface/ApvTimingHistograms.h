#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/ApvTimingSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class ApvTimingHistograms : public CommissioningHistograms {

 public:
  
  ApvTimingHistograms( MonitorUserInterface* );
  ApvTimingHistograms( DaqMonitorBEInterface* );
  virtual ~ApvTimingHistograms();
  
  typedef SummaryPlotFactory<ApvTimingAnalysis*> Factory;
  typedef std::map<uint32_t,ApvTimingAnalysis*> Analyses;
  
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
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H

