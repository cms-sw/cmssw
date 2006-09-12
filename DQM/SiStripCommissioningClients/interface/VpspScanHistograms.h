#ifndef DQM_SiStripCommissioningClients_VpspScanHistograms_H
#define DQM_SiStripCommissioningClients_VpspScanHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/VpspScanSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/VpspScanAnalysis.h"

class MonitorUserInterface;

class VpspScanHistograms : public CommissioningHistograms {

 public:
  
  VpspScanHistograms( MonitorUserInterface* );
  virtual ~VpspScanHistograms();
  
  typedef SummaryHistogramFactory<VpspScanAnalysis> Factory;
  
  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::SummaryHisto&,
			   const sistrip::SummaryType&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );

 protected:

  std::map<uint32_t,VpspScanAnalysis> data_;

  std::auto_ptr<Factory> factory_;

};

#endif // DQM_SiStripCommissioningClients_VpspScanHistograms_H


