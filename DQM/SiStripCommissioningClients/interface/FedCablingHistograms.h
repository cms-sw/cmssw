#ifndef DQM_SiStripCommissioningClients_FedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/FedCablingSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/FedCablingAnalysis.h"


class DQMStore;

class FedCablingHistograms : virtual public CommissioningHistograms {

 public:
  
  FedCablingHistograms( const edm::ParameterSet& pset, DQMStore* );
  ~FedCablingHistograms() override;
  
  typedef SummaryPlotFactory<FedCablingAnalysis*> Factory;
  typedef std::map<uint32_t,FedCablingAnalysis*> Analyses;

  /** */
  void histoAnalysis( bool debug ) override;

  /** */
  void printAnalyses() override;
  
  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& ) override;
  
 protected: 
  
  Analyses data_;
  
  std::unique_ptr<Factory> factory_;

};

#endif // DQM_SiStripCommissioningClients_FedCablingHistograms_H


