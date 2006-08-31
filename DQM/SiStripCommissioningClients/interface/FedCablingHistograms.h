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
  
  typedef SummaryHistogramFactory<FedCablingAnalysis::Monitorables> Factory;

  /** */
  void histoAnalysis();
  /** */
  void createSummaryHisto( const sistrip::SummaryHisto&, 
			   const sistrip::SummaryType&, 
			   const std::string& directory );

 protected: 
  
  std::map<uint32_t,FedCablingAnalysis::Monitorables> data_;
  
  std::auto_ptr<Factory> factory_;

};

#endif // DQM_SiStripCommissioningClients_FedCablingHistograms_H


