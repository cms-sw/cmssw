/****************************************************************************
*
* This is a part of PPS-TOTEM offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#ifndef RecoPPS_Local_TimingRecHitProducerAlgorithm
#define RecoPPS_Local_TimingRecHitProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibrationLUT.h"

/// \tparam G Geometry condition format definition
/// \tparam D Digis input collection type
/// \tparam R RecHits output collection type
template <typename G, typename D, typename R>
class TimingRecHitProducerAlgorithm {
public:
  explicit TimingRecHitProducerAlgorithm(const edm::ParameterSet& iConfig)
      : ts_to_ns_(iConfig.getParameter<double>("timeSliceNs")),
        apply_calib_(iConfig.getParameter<bool>("applyCalibration")) {}
  virtual ~TimingRecHitProducerAlgorithm() = default;

  void setCalibration(const PPSTimingCalibration& calib, const PPSTimingCalibrationLUT& calibLUT) {
    calib_ = &calib;
    calibLUT_ = &calibLUT;
    calib_fct_ = std::make_unique<reco::FormulaEvaluator>(calib_->formula());
  }
  virtual void build(const G&, const D&, R&) = 0;

protected:
  /// Conversion constant between time slice and absolute time (in ns)
  double ts_to_ns_;
  /// Switch on/off the timing calibration
  bool apply_calib_;
  /// DB-loaded calibration object
  const PPSTimingCalibration* calib_{nullptr};
  const PPSTimingCalibrationLUT* calibLUT_{nullptr};
  /// Timing correction formula
  std::unique_ptr<reco::FormulaEvaluator> calib_fct_;
};

#endif
