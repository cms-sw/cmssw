#ifndef DQM_SiStripCommissioningClients_ApvTimingHistograms_H
#define DQM_SiStripCommissioningClients_ApvTimingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"
#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"

class MonitorUserInterface;
class SiStripSummary;

class ApvTimingHistograms : public CommissioningHistograms {

 public:
  
  typedef SummaryHistogramFactory<ApvTimingAnalysis::Monitorables> Factory;

  /** */
  ApvTimingHistograms( MonitorUserInterface* );
  /** */
  virtual ~ApvTimingHistograms();
  
  /** */
  void histoAnalysis();

  /** */
  void createSummaryHisto( const sistrip::SummaryHisto&, 
			   const sistrip::SummaryType&, 
			   const std::string& directory );

 private:

  std::map<uint32_t,ApvTimingAnalysis::Monitorables> data_;

  std::auto_ptr<Factory> factory_;
  
};

#endif // DQM_SiStripCommissioningClients_ApvTimingHistograms_H

