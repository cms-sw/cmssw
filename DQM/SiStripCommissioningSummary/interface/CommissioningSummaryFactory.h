#ifndef DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
#define DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"

class CommissioningAnalysis;

template<>
class SummaryPlotFactory<CommissioningAnalysis*> : public SummaryPlotFactoryBase {
  
 public:
  
  SummaryPlotFactory<CommissioningAnalysis*>() {;}
  virtual ~SummaryPlotFactory<CommissioningAnalysis*>() {;}
  
  virtual uint32_t init( const sistrip::Monitorable&, 
			 const sistrip::Presentation&,
			 const sistrip::View&, 
			 const std::string& top_level_dir, 
			 const sistrip::Granularity&,
			 const std::map<uint32_t,CommissioningAnalysis*>& data ) { return 0; }
  
  virtual void fill( TH1& summary_histo ) {;} 
  
};

#endif // DQM_SiStripCommissioningSummary_CommissioningSummaryFactory_H
