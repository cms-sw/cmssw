#ifndef DQM_SiStripCommissioningClients_FedTimingHistograms_H
#define DQM_SiStripCommissioningClients_FedTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/FedTimingSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FedTimingAnalysis.h"

class MonitorUserInterface;
class SiStripSummary;

class FedTimingHistograms : public CommissioningHistograms {

 public:
  
  FedTimingHistograms( MonitorUserInterface* );
  virtual ~FedTimingHistograms();

  typedef SummaryHistogramFactory<FedTimingAnalysis::Monitorables> Factory;
  
  /** */
  void histoAnalysis();
  /** */
  void createSummaryHisto( const sistrip::SummaryHisto&, 
			   const sistrip::SummaryType&, 
			   const std::string& directory );

 private:

  std::map<uint32_t,FedTimingAnalysis::Monitorables> data_;

  std::auto_ptr<Factory> factory_;
  
};

#endif // DQM_SiStripCommissioningClients_FedTimingHistograms_H

