#ifndef DQM_SiStripCommissioningClients_PedestalsHistograms_H
#define DQM_SiStripCommissioningClients_PedestalsHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/PedestalsSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class PedestalsHistograms : public CommissioningHistograms {

 public:
  
  PedestalsHistograms( MonitorUserInterface* );
  PedestalsHistograms( DaqMonitorBEInterface* );
  virtual ~PedestalsHistograms();

  typedef SummaryPlotFactory<PedestalsAnalysis*> Factory;
  typedef std::map<uint32_t,PedestalsAnalysis*> Analyses;
  
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

#endif // DQM_SiStripCommissioningClients_PedestalsHistograms_H
