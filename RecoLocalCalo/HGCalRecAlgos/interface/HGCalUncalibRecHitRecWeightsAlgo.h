#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalUncalibRecHitRecWeightsAlgo_HH
#define RecoLocalCalo_HGCalRecAlgos_HGCalUncalibRecHitRecWeightsAlgo_HH

/** \class HGalUncalibRecHitRecWeightsAlgo
  *  compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method, a la Ecal 
  *
  *  \author Valeri Andreev
  *  
  *
  */

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalUncalibRecHitRecAbsAlgo.h"
#include <vector>
#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

template <class C>
class HGCalUncalibRecHitRecWeightsAlgo {
public:
  // destructor
  virtual ~HGCalUncalibRecHitRecWeightsAlgo(){};

  void set_isSiFESim(const bool isSiFE) { isSiFESim_ = isSiFE; }
  bool isSiFESim() const { return isSiFESim_; }

  void set_ADCLSB(const double adclsb) { adcLSB_ = adclsb; }
  void set_TDCLSB(const double tdclsb) { tdcLSB_ = tdclsb; }

  void set_toaLSBToNS(const double lsb2ns) { toaLSBToNS_ = lsb2ns; }

  void set_tdcOnsetfC(const double tdcOnset) { tdcOnsetfC_ = tdcOnset; }

  void set_tofDelay(const double tofDelay) { tofDelay_ = tofDelay; }

  void set_fCPerMIP(const std::vector<double>& fCPerMIP) {
    if (std::any_of(fCPerMIP.cbegin(), fCPerMIP.cend(), [](double conv) { return conv <= 0.0; })) {
      throw cms::Exception("BadConversionFactor") << "At least one of fCPerMIP is zero!" << std::endl;
    }
    fCPerMIP_ = fCPerMIP;
  }

  bool setGeometry(const edm::ESHandle<HGCalGeometry>& geom) {
    geom_ = geom.isValid() ? geom.product() : nullptr;
    if (isSiFESim_ && geom_)
      ddd_ = &(geom_->topology().dddConstants());
    else
      ddd_ = nullptr;
    return geom_ != nullptr;
  }

  /// Compute HGCUncalibratedRecHit from DataFrame
  virtual HGCUncalibratedRecHit makeRecHit(const C& dataFrame) {
    double amplitude_(-1.), pedestal_(-1.), jitter_(-99.), chi2_(-1.);
    uint32_t flag = 0;

    constexpr int iSample = 2;  //only in-time sample
    const auto& sample = dataFrame.sample(iSample);

    // Were digis done w/ the complete digitization and using signal shape?
    // (originally done only for the silicon, while for scitillator it was trivial. Fomr 11_ also scinti uses shape)
    if (isSiFESim_) {
      // mode == true: TDC readout was activated and amplitude comes from TimeOverThreshold
      if (sample.mode()) {
        flag = !sample.threshold();  // raise flag if busy cell
        // LG (23/06/2015):
        // to get a continuous energy spectrum we must add here the maximum value in fC ever
        // reported by the ADC. Namely: floor(tdcOnset/adcLSB_) * adcLSB_
        // need to increment by one so TDC doesn't overlap with ADC last bin
        // LG (11/04/2016):
        // offset the TDC upwards to reflect the bin center
        amplitude_ = (std::floor(tdcOnsetfC_ / adcLSB_) + 1.0) * adcLSB_ + (double(sample.data()) + 0.5) * tdcLSB_;
      } else {
        amplitude_ = double(sample.data()) * adcLSB_;  // why do we not have +0.5 here ?
      }                                                //isSiFESim_
    }                                                  //mode()

    // trivial digitization, i.e. no signal shape
    else {
      amplitude_ = double(sample.data()) * adcLSB_;
    }

    if (sample.getToAValid()) {
      const auto& dist2center = geom_ ? geom_->getPosition(dataFrame.id()).mag() : 0;
      jitter_ = double(sample.toa()) * toaLSBToNS_ - dist2center / c_cm_ns - tofDelay_;
    }

    int thickness = (ddd_ != nullptr) ? ddd_->waferType(dataFrame.id(), false) : 0;
    amplitude_ = amplitude_ / fCPerMIP_[thickness];

    LogDebug("HGCUncalibratedRecHit") << "isSiFESim_: " << isSiFESim_ << " ADC+: set the charge to: " << amplitude_
                                      << ' ' << sample.data() << ' ' << adcLSB_ << ' '
                                      << "               TDC+: set the ToA to: " << jitter_ << ' ' << sample.toa()
                                      << ' ' << toaLSBToNS_ << ' ' << tdcLSB_
                                      << "               getToAValid(): " << sample.getToAValid()
                                      << " mode(): " << sample.mode() << std::endl;
    LogDebug("HGCUncalibratedRecHit") << "Final uncalibrated amplitude : " << amplitude_ << std::endl;

    return HGCUncalibratedRecHit(dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_, flag);
  }

private:
  static constexpr float c_cm_ns = CLHEP::c_light * CLHEP::ns / CLHEP::cm;
  double adcLSB_, tdcLSB_, toaLSBToNS_, tdcOnsetfC_, tofDelay_;
  bool isSiFESim_;
  std::vector<double> fCPerMIP_;
  const HGCalDDDConstants* ddd_;
  const HGCalGeometry* geom_;
};
#endif
