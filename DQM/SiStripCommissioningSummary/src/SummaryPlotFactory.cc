#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
template <class T>
uint32_t SummaryPlotFactory<T>::init(const sistrip::Monitorable& mon,
                                     const sistrip::Presentation& pres,
                                     const sistrip::View& view,
                                     const std::string& level,
                                     const sistrip::Granularity& gran,
                                     const std::map<uint32_t, T>& data) {
  // Some initialisation
  SummaryPlotFactoryBase::init(mon, pres, view, level, gran);

  // Transfer appropriate monitorables info to generator object
  if (!SummaryPlotFactoryBase::generator_) {
    return 0;
  }
  typename std::map<uint32_t, T>::const_iterator iter = data.begin();
  for (; iter != data.end(); iter++) {
    static float value = static_cast<float>(iter->second);
    SummaryPlotFactoryBase::generator_->fillMap(
        SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, iter->first, value);
  }

  return SummaryPlotFactoryBase::generator_->nBins();
}

// -----------------------------------------------------------------------------
//
template <class T>
void SummaryPlotFactory<T>::fill(TH1& summary_histo) {
  // Some initialisation
  SummaryPlotFactoryBase::fill(summary_histo);
}

// -----------------------------------------------------------------------------
//
template class SummaryPlotFactory<uint32_t>;
template class SummaryPlotFactory<uint16_t>;
template class SummaryPlotFactory<int32_t>;
template class SummaryPlotFactory<int16_t>;
template class SummaryPlotFactory<double>;
template class SummaryPlotFactory<float>;
