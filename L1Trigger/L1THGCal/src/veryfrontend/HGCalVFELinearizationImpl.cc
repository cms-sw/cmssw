#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFELinearizationImpl.h"

HGCalVFELinearizationImpl::HGCalVFELinearizationImpl(const edm::ParameterSet& conf)
    : linLSB_si_(conf.getParameter<double>("linLSB_si")),
      adcsaturation_si_(conf.getParameter<double>("adcsaturation_si")),
      tdcnBits_si_(conf.getParameter<uint32_t>("tdcnBits_si")),
      tdcOnset_si_(conf.getParameter<double>("tdcOnset_si")),
      adcnBits_si_(conf.getParameter<uint32_t>("adcnBits_si")),
      tdcsaturation_si_(conf.getParameter<double>("tdcsaturation_si")),
      linLSB_sc_(conf.getParameter<double>("linLSB_sc")),
      adcsaturation_sc_(conf.getParameter<double>("adcsaturation_sc")),
      tdcnBits_sc_(conf.getParameter<uint32_t>("tdcnBits_sc")),
      tdcOnset_sc_(conf.getParameter<double>("tdcOnset_sc")),
      adcnBits_sc_(conf.getParameter<uint32_t>("adcnBits_sc")),
      tdcsaturation_sc_(conf.getParameter<double>("tdcsaturation_sc")),
      linnBits_(conf.getParameter<uint32_t>("linnBits")) {
  adcLSB_si_ = adcsaturation_si_ / pow(2., adcnBits_si_);
  tdcLSB_si_ = tdcsaturation_si_ / pow(2., tdcnBits_si_);
  adcLSB_sc_ = adcsaturation_sc_ / pow(2., adcnBits_sc_);
  tdcLSB_sc_ = tdcsaturation_sc_ / pow(2., tdcnBits_sc_);
  linMax_ = (0x1 << linnBits_) - 1;
}

void HGCalVFELinearizationImpl::linearize(const std::vector<HGCDataFrame<DetId, HGCSample>>& dataframes,
                                          std::vector<std::pair<DetId, uint32_t>>& linearized_dataframes) {
  double amplitude = 0.;
  uint32_t amplitude_int = 0;
  const int kIntimeSample = 2;

  for (const auto& frame : dataframes) {  //loop on DIGI
    unsigned det = frame.id().det();
    double adcLSB = 0.;
    double tdcLSB = 0.;
    double tdcOnset = 0.;
    double linLSB = 0.;
    if (det == DetId::Forward || det == DetId::HGCalEE || det == DetId::HGCalHSi) {
      adcLSB = adcLSB_si_;
      tdcLSB = tdcLSB_si_;
      tdcOnset = tdcOnset_si_;
      linLSB = linLSB_si_;
    } else if (det == DetId::Hcal || det == DetId::HGCalHSc) {
      adcLSB = adcLSB_sc_;
      tdcLSB = tdcLSB_sc_;
      tdcOnset = tdcOnset_sc_;
      linLSB = linLSB_sc_;
    }
    if (frame[kIntimeSample].mode()) {  //TOT mode
      amplitude = (floor(tdcOnset / adcLSB) + 1.0) * adcLSB + double(frame[kIntimeSample].data()) * tdcLSB;
    } else {  //ADC mode
      amplitude = double(frame[kIntimeSample].data()) * adcLSB;
    }

    amplitude_int = uint32_t(floor(amplitude / linLSB + 0.5));
    if (amplitude_int > linMax_)
      amplitude_int = linMax_;

    linearized_dataframes.push_back(std::make_pair(frame.id(), amplitude_int));
  }
}
