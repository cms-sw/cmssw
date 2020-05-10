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
      linnBits_(conf.getParameter<uint32_t>("linnBits")),
      oot_coefficients_(conf.getParameter<std::vector<double>>("oot_coefficients")) {
  constexpr int kOot_order = 2;
  if (oot_coefficients_.size() != kOot_order) {
    throw cms::Exception("BadConfiguration") << "OOT subtraction needs " << kOot_order << " coefficients";
  }
  adcLSB_si_ = ldexp(adcsaturation_si_, -adcnBits_si_);
  tdcLSB_si_ = ldexp(tdcsaturation_si_, -tdcnBits_si_);
  adcLSB_sc_ = ldexp(adcsaturation_sc_, -adcnBits_sc_);
  tdcLSB_sc_ = ldexp(tdcsaturation_sc_, -tdcnBits_sc_);
  linMax_ = (0x1 << linnBits_) - 1;
}

void HGCalVFELinearizationImpl::linearize(const std::vector<HGCDataFrame<DetId, HGCSample>>& dataframes,
                                          std::vector<std::pair<DetId, uint32_t>>& linearized_dataframes) {
  constexpr int kIntimeSample = 2;
  constexpr int kOuttime1Sample = 1;  // in time - 1;
  constexpr int kOuttime2Sample = 0;  // in time - 2;

  for (const auto& frame : dataframes) {  //loop on DIGI
    double amplitude = 0.;
    uint32_t amplitude_int = 0;
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
      if (frame[kIntimeSample].mode()) {  //TOT mode
        amplitude = (floor(tdcOnset / adcLSB) + 1.0) * adcLSB + double(frame[kIntimeSample].data()) * tdcLSB;
      } else {  //ADC mode
        double data = frame[kIntimeSample].data();
        // applies OOT PU subtraction only in the ADC mode
        if (!frame[kOuttime1Sample].mode()) {
          data += oot_coefficients_[kOuttime1Sample] * frame[kOuttime1Sample].data();
          if (!frame[kOuttime2Sample].mode()) {
            data += oot_coefficients_[kOuttime2Sample] * frame[kOuttime2Sample].data();
          }
        }
        amplitude = std::max(0., data) * adcLSB;
      }

      amplitude_int = uint32_t(floor(amplitude / linLSB + 0.5));
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
    if (amplitude_int == 0)
      continue;
    if (amplitude_int > linMax_)
      amplitude_int = linMax_;

    linearized_dataframes.push_back(std::make_pair(frame.id(), amplitude_int));
  }
}
