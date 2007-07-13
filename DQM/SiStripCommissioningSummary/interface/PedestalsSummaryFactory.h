#ifndef DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H
#define DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"
#include "DQM/SiStripCommissioningAnalysis/interface/PedestalsAnalysis.h"

template<>
class SummaryPlotFactory<PedestalsAnalysis*> : public SummaryPlotFactoryBase {
  
 public:
  
  uint32_t init( const sistrip::Monitorable&, 
		 const sistrip::Presentation&,
		 const sistrip::View&, 
		 const std::string& top_level_dir, 
		 const sistrip::Granularity&,
		 const std::map<uint32_t,PedestalsAnalysis*>& data );
  
  void fill( TH1& summary_histo );

};

#endif // DQM_SiStripCommissioningSummary_PedestalsSummaryFactory_H
