#ifndef DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H
#define DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H

#include "DQM/SiStripCommissioningSummary/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommissioningAnalysis/interface/OptoScanAnalysis.h"

class SummaryGenerator;

template<>
class SummaryHistogramFactory<OptoScanAnalysis> {
  
 public:
  
  SummaryHistogramFactory();
  ~SummaryHistogramFactory();

  void init( const sistrip::SummaryHisto&, 
	     const sistrip::SummaryType&,
	     const sistrip::View&, 
	     const std::string& top_level_dir, 
	     const sistrip::Granularity& );
  
  uint32_t extract( const std::map<uint32_t,OptoScanAnalysis>& data );
  
  void fill( TH1& summary_histo );
  
 private:
  
  sistrip::SummaryHisto histo_;
  sistrip::SummaryType type_;
  sistrip::View view_;
  std::string level_;
  sistrip::Granularity gran_;
  SummaryGenerator* generator_;

};

#endif // DQM_SiStripCommissioningSummary_OptoScanSummaryFactory_H
