#ifndef DQM_SiStripCommissioningClients_FedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/FedCablingSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FedCablingAnalysis.h"

class MonitorUserInterface;

class FedCablingHistograms : public CommissioningHistograms {

 public:
  
  FedCablingHistograms( MonitorUserInterface* );
  virtual ~FedCablingHistograms();
  
  typedef SummaryHistogramFactory<FedCablingAnalysis> Factory;

  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );

 protected: 
  
  std::map<uint32_t,FedCablingAnalysis> data_;
  
  std::auto_ptr<Factory> factory_;

};

#endif // DQM_SiStripCommissioningClients_FedCablingHistograms_H


