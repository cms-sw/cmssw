#ifndef DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H
#define DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FastFedCablingAnalysis.h"

template<>
class SummaryPlotFactory<FastFedCablingAnalysis*> : public SummaryPlotFactoryBase {
  
 public:
  
  uint32_t init( const sistrip::Monitorable&, 
		 const sistrip::Presentation&,
		 const sistrip::View&, 
		 const std::string& top_level_dir, 
		 const sistrip::Granularity&,
		 const std::map<uint32_t,FastFedCablingAnalysis*>& data );
  
  void fill( TH1& summary_histo );
  
};

#endif // DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H
