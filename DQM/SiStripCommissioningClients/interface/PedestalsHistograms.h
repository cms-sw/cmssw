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

  typedef SummaryHistogramFactory<PedestalsAnalysis::Monitorables> Factory;
  
  /** */
  void histoAnalysis();
  /** */
  void createSummaryHisto( const sistrip::SummaryHisto&, 
			   const sistrip::SummaryType&, 
			   const std::string& directory );

 private:

  std::map<uint32_t,PedestalsAnalysis::Monitorables> data_;

  std::auto_ptr<Factory> factory_;

};

#endif // DQM_SiStripCommissioningClients_PedestalsHistograms_H
