#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"

// -----------------------------------------------------------------------------
//
uint32_t SummaryPlotFactory<CommissioningAnalysis*>::init(const sistrip::Monitorable& mon,
                                                          const sistrip::Presentation& pres,
                                                          const sistrip::View& view,
                                                          const std::string& level,
                                                          const sistrip::Granularity& gran,
                                                          const std::map<uint32_t, CommissioningAnalysis*>& data) {
  // initialisation
  SummaryPlotFactoryBase::init(mon, pres, view, level, gran);

  // check if generator object exists
  if (!SummaryPlotFactoryBase::generator_) {
    return 0;
  }

  // extract monitorables and populate map
  std::map<uint32_t, CommissioningAnalysis*>::const_iterator iter = data.begin();
  for (; iter != data.end(); iter++) {
    extract(iter);
  }

  return SummaryPlotFactoryBase::generator_->nBins();
}

// -----------------------------------------------------------------------------
//
void SummaryPlotFactory<CommissioningAnalysis*>::fill(TH1& summary_histo) {
  // fill histogram
  SummaryPlotFactoryBase::fill(summary_histo);

  // check if generator object exists
  if (!SummaryPlotFactoryBase::generator_) {
    return;
  }

  // format histo
  format();
}
