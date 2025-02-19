#ifndef DQM_SiStripCommissioningSummary_SummaryPlotFactoryBase_H
#define DQM_SiStripCommissioningSummary_SummaryPlotFactoryBase_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "TH1.h"
#include <string>

class SummaryPlotFactoryBase {
  
 protected:
    
  void init( const sistrip::Monitorable&, 
	     const sistrip::Presentation&,
	     const sistrip::View&, 
	     const std::string& top_level_dir, 
	     const sistrip::Granularity& );
  
  void fill( TH1& summary_histo );

 protected:

  // Constructors, destructors
  SummaryPlotFactoryBase();
  ~SummaryPlotFactoryBase();
  
  // Parameters
  sistrip::Monitorable mon_;
  sistrip::Presentation pres_;
  sistrip::View view_;
  std::string level_;
  sistrip::Granularity gran_;
  
  // Summary plot generator class
  SummaryGenerator* generator_;
  
};

#endif // DQM_SiStripCommissioningSummary_SummaryPlotFactoryBase_H



