#ifndef DQM_SiStripCommissioningClients_OptoScanHistograms_H
#define DQM_SiStripCommissioningClients_OptoScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/OptoScanSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/OptoScanAnalysis.h"

class MonitorUserInterface;

class OptoScanHistograms : public CommissioningHistograms {

 public:

  OptoScanHistograms( MonitorUserInterface* );
  virtual ~OptoScanHistograms();

  typedef SummaryHistogramFactory<OptoScanAnalysis::Monitorables> Factory;
  
  /** */
  void histoAnalysis();
  /** */
  void createSummaryHisto( const sistrip::SummaryHisto&, 
			   const sistrip::SummaryType&, 
			   const std::string& directory );

 protected:

  std::map<uint32_t,OptoScanAnalysis::Monitorables> data_;

  std::auto_ptr<Factory> factory_;

};

#endif // DQM_SiStripCommissioningClients_OptoScanHistograms_H


