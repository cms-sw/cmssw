#ifndef DQM_SiStripCommissioningClients_PedestalsHistograms_H
#define DQM_SiStripCommissioningClients_PedestalsHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/PedestalsSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"

class MonitorUserInterface;

class PedestalsHistograms : public CommissioningHistograms {

 public:
  
  PedestalsHistograms( MonitorUserInterface* );
  virtual ~PedestalsHistograms();

  typedef SummaryHistogramFactory<PedestalsAnalysis> Factory;
  
  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );

 protected:

  std::map<uint32_t,PedestalsAnalysis> data_;

  std::auto_ptr<Factory> factory_;

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistograms_H
