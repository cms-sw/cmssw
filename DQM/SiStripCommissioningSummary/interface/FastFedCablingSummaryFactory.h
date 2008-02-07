#ifndef DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H
#define DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H

/* #include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactory.h" */
/* #include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h" */
/* #include "CondFormats/SiStripObjects/interface/FastFedCablingAnalysis.h" */

#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"

class CommissioningAnalysis;

class FastFedCablingSummaryFactory : public SummaryPlotFactory<CommissioningAnalysis*> {
  
 public:
  
  uint32_t init( const sistrip::Monitorable&, 
		 const sistrip::Presentation&,
		 const sistrip::View&, 
		 const std::string& top_level_dir, 
		 const sistrip::Granularity&,
		 const std::map<uint32_t,CommissioningAnalysis*>& data );
  
  void fill( TH1& summary_histo );
  
};

/* template<> */
/* class SummaryPlotFactory<FastFedCablingAnalysis*> : public SummaryPlotFactoryBase { */
  
/*  public: */
  
/*   uint32_t init( const sistrip::Monitorable&,  */
/* 		 const sistrip::Presentation&, */
/* 		 const sistrip::View&,  */
/* 		 const std::string& top_level_dir,  */
/* 		 const sistrip::Granularity&, */
/* 		 const std::map<uint32_t,FastFedCablingAnalysis*>& data ); */
  
/*   void fill( TH1& summary_histo ); */
  
/* }; */

#endif // DQM_SiStripCommissioningSummary_FastFedCablingSummaryFactory_H
