#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class FastFedCablingHistograms : public virtual CommissioningHistograms {

 public:
  
  FastFedCablingHistograms( MonitorUserInterface* );
  FastFedCablingHistograms( DaqMonitorBEInterface* );
  virtual ~FastFedCablingHistograms();
  
  void histoAnalysis( bool debug );
  
  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );
  
};

#endif // DQM_SiStripCommissioningClients_FastFedCablingHistograms_H

