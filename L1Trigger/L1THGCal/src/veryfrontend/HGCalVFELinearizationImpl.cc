#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFELinearizationImpl.h"

HGCalVFELinearizationImpl::HGCalVFELinearizationImpl(const edm::ParameterSet& conf)
    : linLSB_(conf.getParameter<double>("linLSB")),
      adcsaturation_(conf.getParameter<double>("adcsaturation")),
      tdcnBits_(conf.getParameter<uint32_t>("tdcnBits")),
      tdcOnsetfC_(conf.getParameter<double>("tdcOnsetfC")),
      adcnBits_(conf.getParameter<uint32_t>("adcnBits")),
      tdcsaturation_(conf.getParameter<double>("tdcsaturation")),
      linnBits_(conf.getParameter<uint32_t>("linnBits")) {
  adcLSB_ = adcsaturation_ / pow(2., adcnBits_);
  tdcLSB_ = tdcsaturation_ / pow(2., tdcnBits_);
  linMax_ = (0x1 << linnBits_) - 1;
}

void HGCalVFELinearizationImpl::linearize(const std::vector<HGCDataFrame<DetId, HGCSample>>& dataframes,
                                          std::vector<std::pair<DetId, uint32_t>>& linearized_dataframes) {
  double amplitude = 0.;
  uint32_t amplitude_int = 0;
  const int kIntimeSample = 2;

  for (const auto& frame : dataframes) {  //loop on DIGI
    unsigned det = frame.id().det();
    if (det == DetId::Forward || det == DetId::HGCalEE || det == DetId::HGCalHSi) {
      if (frame[kIntimeSample].mode()) {  //TOT mode
        amplitude = (floor(tdcOnsetfC_ / adcLSB_) + 1.0) * adcLSB_ + double(frame[kIntimeSample].data()) * tdcLSB_;
      } else {  //ADC mode
        amplitude = double(frame[kIntimeSample].data()) * adcLSB_;
      }

      amplitude_int = uint32_t(floor(amplitude / linLSB_ + 0.5));
    } else if (det == DetId::Hcal || det == DetId::HGCalHSc) {
      // no linearization here. Take the raw ADC data
      amplitude_int = frame[kIntimeSample].data();
    }
    if (amplitude_int > linMax_)
      amplitude_int = linMax_;

    linearized_dataframes.push_back(std::make_pair(frame.id(), amplitude_int));
  }
}
