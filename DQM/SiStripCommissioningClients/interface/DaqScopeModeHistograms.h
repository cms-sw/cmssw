#ifndef DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H
#define DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/DaqScopeModeSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/DaqScopeModeAnalysis.h"


class DQMStore;

class DaqScopeModeHistograms : public CommissioningHistograms {

 public:
  
  DaqScopeModeHistograms( const edm::ParameterSet& pset, DQMStore* );
  virtual ~DaqScopeModeHistograms();
  
  typedef SummaryHistogramFactory<DaqScopeModeAnalysis> Factory;
  
  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );
  
 protected: 
  
  std::map<uint32_t,DaqScopeModeAnalysis> data_;
  
  std::auto_ptr<Factory> factory_;
  
};

#endif // DQM_SiStripCommissioningClients_DaqScopeModeHistograms_H

