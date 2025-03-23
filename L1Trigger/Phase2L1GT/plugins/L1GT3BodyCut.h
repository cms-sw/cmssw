#ifndef L1Trigger_Phase2L1GT_L1GT3BodyCut_h
#define L1Trigger_Phase2L1GT_L1GT3BodyCut_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTInvariantMassError.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"

#include "L1GTSingleInOutLUT.h"
#include "L1GTOptionalParam.h"

#include <optional>
#include <cinttypes>

namespace l1t {

  class L1GT3BodyCut {
  public:
    L1GT3BodyCut(const edm::ParameterSet& config,
                 const edm::ParameterSet& lutConfig,
                 const L1GTScales& scales,
                 bool inv_mass_checks = false)
        : scales_(scales),
          coshEtaLUT_(lutConfig.getParameterSet("cosh_eta_lut")),
          coshEtaLUT2_(lutConfig.getParameterSet("cosh_eta_lut2")),
          cosPhiLUT_(lutConfig.getParameterSet("cos_phi_lut")),
          minInvMassSqrDiv2_(getOptionalParam<int64_t, double>("minInvMass",
                                                               config,
                                                               [&](double value) {
                                                                 return std::floor(scales.to_hw_InvMassSqrDiv2(value) *
                                                                                   cosPhiLUT_.output_scale());
                                                               })),
          maxInvMassSqrDiv2_(getOptionalParam<int64_t, double>(
              "maxInvMass",
              config,
              [&](double value) { return std::ceil(scales.to_hw_InvMassSqrDiv2(value) * cosPhiLUT_.output_scale()); })),
          minTransMassSqrDiv2_(getOptionalParam<int64_t, double>(
              "minTransMass",
              config,
              [&](double value) {
                return std::floor(scales.to_hw_TransMassSqrDiv2(value) * cosPhiLUT_.output_scale());
              })),
          maxTransMassSqrDiv2_(getOptionalParam<int64_t, double>(
              "maxTransMass",
              config,
              [&](double value) {
                return std::ceil(scales.to_hw_TransMassSqrDiv2(value) * cosPhiLUT_.output_scale());
              })),
          scaleNormalShift_(std::round(std::log2(std::ceil(coshEtaLUT_.output_scale() / coshEtaLUT2_.output_scale())))),
          invMassResolutionReduceShift_([&]() {
            // Computation of the dynamic input two-body mass resolution w.r.t. the cut value.
            // The result is a resolution of inputs between 2^-15 to 2^-16 of the cut value.
            if (minInvMassSqrDiv2_) {
              return std::max<int>(std::floor(std::log2(minInvMassSqrDiv2_.value())) + 1 - CALC_BITS, 0);
            } else if (maxInvMassSqrDiv2_) {
              return std::max<int>(std::floor(std::log2(maxInvMassSqrDiv2_.value())) + 1 - CALC_BITS, 0);
            } else {
              return 0;
            }
          }()),
          transMassResolutionReduceShift_([&]() {
            // Computation of the dynamic input two-body mass resolution w.r.t. the cut value.
            // The result is a resolution of inputs between 2^-15 to 2^-16 of the cut value.
            if (minTransMassSqrDiv2_) {
              return std::max<int>(std::floor(std::log2(minTransMassSqrDiv2_.value())) + 1 - CALC_BITS, 0);
            } else if (maxTransMassSqrDiv2_) {
              return std::max<int>(std::floor(std::log2(maxTransMassSqrDiv2_.value())) + 1 - CALC_BITS, 0);
            } else {
              return 0;
            }
          }()),
          inv_mass_checks_(inv_mass_checks) {}

    bool checkObjects(const P2GTCandidate& obj1,
                      const P2GTCandidate& obj2,
                      const P2GTCandidate& obj3,
                      InvariantMassErrorCollection& massErrors) const {
      bool res = true;

      if (minInvMassSqrDiv2_ || maxInvMassSqrDiv2_) {
        int64_t invMassSqrDiv2 = (calc2BodyInvMass(obj1, obj2, massErrors) >> invMassResolutionReduceShift_) +
                                 (calc2BodyInvMass(obj1, obj3, massErrors) >> invMassResolutionReduceShift_) +
                                 (calc2BodyInvMass(obj2, obj3, massErrors) >> invMassResolutionReduceShift_);

        res &= minInvMassSqrDiv2_ ? invMassSqrDiv2 > minInvMassSqrDiv2_.value() >> invMassResolutionReduceShift_ : true;
        res &= maxInvMassSqrDiv2_ ? invMassSqrDiv2 < maxInvMassSqrDiv2_.value() >> invMassResolutionReduceShift_ : true;
      }

      if (minTransMassSqrDiv2_ || maxTransMassSqrDiv2_) {
        int64_t transMassDiv2 = (calc2BodyTransMass(obj1, obj2) >> transMassResolutionReduceShift_) +
                                (calc2BodyTransMass(obj1, obj3) >> transMassResolutionReduceShift_) +
                                (calc2BodyTransMass(obj2, obj3) >> transMassResolutionReduceShift_);

        res &= minTransMassSqrDiv2_ ? transMassDiv2 > minTransMassSqrDiv2_.value() >> transMassResolutionReduceShift_
                                    : true;
        res &= maxTransMassSqrDiv2_ ? transMassDiv2 < maxTransMassSqrDiv2_.value() >> transMassResolutionReduceShift_
                                    : true;
      }

      return res;
    }

    static void fillPSetDescription(edm::ParameterSetDescription& desc) {
      desc.addOptional<double>("minInvMass");
      desc.addOptional<double>("maxInvMass");
      desc.addOptional<double>("minTransMass");
      desc.addOptional<double>("maxTransMass");
    }

  private:
    static constexpr int HW_PI = 1 << (P2GTCandidate::hwPhi_t::width - 1);  // assumes phi in [-pi, pi)
    static constexpr int CALC_BITS = 16;                                    // Allocate 16 bits to the calculation

    int64_t calc2BodyInvMass(const P2GTCandidate& obj1,
                             const P2GTCandidate& obj2,
                             InvariantMassErrorCollection& massErrors) const {
      uint32_t dEta = (obj1.hwEta() > obj2.hwEta()) ? obj1.hwEta().to_int() - obj2.hwEta().to_int()
                                                    : obj2.hwEta().to_int() - obj1.hwEta().to_int();
      int32_t lutCoshDEta = dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT
                                ? coshEtaLUT_[dEta]
                                : coshEtaLUT2_[dEta - L1GTSingleInOutLUT::DETA_LUT_SPLIT];

      // Ensure dPhi is always the smaller angle, i.e. always between [0, pi]
      uint32_t dPhi = HW_PI - abs(abs(obj1.hwPhi().to_int() - obj2.hwPhi().to_int()) - HW_PI);

      // Optimization: (cos(x + pi) = -cos(x), x in [0, pi))
      int32_t lutCosDPhi = dPhi >= HW_PI ? -cosPhiLUT_[dPhi] : cosPhiLUT_[dPhi];

      int64_t invMassSqrDiv2;

      if (dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT) {
        // dEta [0, 2pi)
        invMassSqrDiv2 = obj1.hwPT().to_int64() * obj2.hwPT().to_int64() * (lutCoshDEta - lutCosDPhi);
      } else {
        // dEta [2pi, 4pi), ignore cos
        invMassSqrDiv2 = obj1.hwPT().to_int64() * obj2.hwPT().to_int64() * lutCoshDEta;
      }

      if (inv_mass_checks_) {
        double precInvMass =
            scales_.pT_lsb() * std::sqrt(2 * obj1.hwPT().to_double() * obj2.hwPT().to_double() *
                                         (std::cosh(dEta * scales_.eta_lsb()) - std::cos(dPhi * scales_.phi_lsb())));

        double lutInvMass =
            scales_.pT_lsb() * std::sqrt(2 * static_cast<double>(invMassSqrDiv2) /
                                         (dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT ? coshEtaLUT_.output_scale()
                                                                                    : coshEtaLUT2_.output_scale()));

        double error = std::abs(precInvMass - lutInvMass);
        massErrors.emplace_back(InvariantMassError{error, error / precInvMass, precInvMass});
      }

      // Normalize scales required due to LUT split in dEta with different scale factors.
      return dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT ? invMassSqrDiv2 : invMassSqrDiv2 << scaleNormalShift_;
    }

    int64_t calc2BodyTransMass(const P2GTCandidate& obj1, const P2GTCandidate& obj2) const {
      // Ensure dPhi is always the smaller angle, i.e. always between [0, pi]
      uint32_t dPhi = HW_PI - abs(abs(obj1.hwPhi().to_int() - obj2.hwPhi().to_int()) - HW_PI);

      // Optimization: (cos(x + pi) = -cos(x), x in [0, pi))
      int32_t lutCosDPhi = dPhi >= HW_PI ? -cosPhiLUT_[dPhi] : cosPhiLUT_[dPhi];

      return obj1.hwPT().to_int64() * obj2.hwPT().to_int64() *
             (static_cast<int64_t>(std::round(cosPhiLUT_.output_scale())) - lutCosDPhi);
    }

    const L1GTScales& scales_;

    const L1GTSingleInOutLUT coshEtaLUT_;   // [0, 2pi)
    const L1GTSingleInOutLUT coshEtaLUT2_;  // [2pi, 4pi)
    const L1GTSingleInOutLUT cosPhiLUT_;

    const std::optional<int64_t> minInvMassSqrDiv2_;
    const std::optional<int64_t> maxInvMassSqrDiv2_;
    const std::optional<int64_t> minTransMassSqrDiv2_;
    const std::optional<int64_t> maxTransMassSqrDiv2_;

    const int scaleNormalShift_;
    const int invMassResolutionReduceShift_;
    const int transMassResolutionReduceShift_;

    const bool inv_mass_checks_;
  };

}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GT3BodyCut_h
