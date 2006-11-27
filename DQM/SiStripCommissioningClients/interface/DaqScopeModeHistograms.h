#ifndef DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H
#define DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/DaqScopeModeSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/DaqScopeModeAnalysis.h"

class MonitorUserInterface;

class DaqScopeModeHistograms : public CommissioningHistograms {

 public:
  
  DaqScopeModeHistograms( MonitorUserInterface* );
  virtual ~DaqScopeModeHistograms();
  
  typedef SummaryHistogramFactory<DaqScopeModeAnalysis> Factory;
  
  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::SummaryHisto&,
			   const sistrip::SummaryType&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );
  
 protected: 
  
  std::map<uint32_t,DaqScopeModeAnalysis> data_;
  
  std::auto_ptr<Factory> factory_;
  
};

#endif // DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H

