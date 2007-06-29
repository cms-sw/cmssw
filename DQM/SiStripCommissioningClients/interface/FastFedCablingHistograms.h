#ifndef DQM_SiStripCommissioningClients_FastFedCablingHistograms_H
#define DQM_SiStripCommissioningClients_FastFedCablingHistograms_H

#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/FastFedCablingSummaryFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FastFedCablingAnalysis.h"

class MonitorUserInterface;
class DaqMonitorBEInterface;

class FastFedCablingHistograms : public CommissioningHistograms {

 public:
  
  FastFedCablingHistograms( MonitorUserInterface* );
  FastFedCablingHistograms( DaqMonitorBEInterface* );
  virtual ~FastFedCablingHistograms();
  
  typedef SummaryPlotFactory<FastFedCablingAnalysis*> Factory;
  typedef std::map<uint32_t,FastFedCablingAnalysis*> Analyses;

  /** */
  void histoAnalysis( bool debug );

  /** */
  void createSummaryHisto( const sistrip::Monitorable&,
			   const sistrip::Presentation&,
			   const std::string& top_level_dir,
			   const sistrip::Granularity& );

 protected: 
  
  Analyses data_;
  
  std::auto_ptr<Factory> factory_;
  
};

#endif // DQM_SiStripCommissioningClients_FastFedCablingHistograms_H


