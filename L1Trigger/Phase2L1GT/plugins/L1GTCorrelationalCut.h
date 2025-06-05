#ifndef L1Trigger_Phase2L1GT_L1GTCorrelationalCut_h
#define L1Trigger_Phase2L1GT_L1GTCorrelationalCut_h

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

  class L1GTCorrelationalCut {
  public:
    L1GTCorrelationalCut(const edm::ParameterSet& config,
                         const edm::ParameterSet& lutConfig,
                         const L1GTScales& scales,
                         bool enable_sanity_checks = false,
                         bool inv_mass_checks = false)
        : scales_(scales),
          coshEtaLUT_(lutConfig.getParameterSet("cosh_eta_lut")),
          coshEtaLUT2_(lutConfig.getParameterSet("cosh_eta_lut2")),
          cosPhiLUT_(lutConfig.getParameterSet("cos_phi_lut")),
          minDEta_(getOptionalParam<int, double>(
              "minDEta", config, [&scales](double value) { return scales.to_hw_eta_floor(value); })),
          maxDEta_(getOptionalParam<int, double>(
              "maxDEta", config, [&scales](double value) { return scales.to_hw_eta_ceil(value); })),
          minDPhi_(getOptionalParam<int, double>(
              "minDPhi", config, [&scales](double value) { return scales.to_hw_phi_floor(value); })),
          maxDPhi_(getOptionalParam<int, double>(
              "maxDPhi", config, [&scales](double value) { return scales.to_hw_phi_ceil(value); })),
          minDz_(getOptionalParam<int, double>(
              "minDz", config, [&scales](double value) { return scales.to_hw_z0_floor(value); })),
          maxDz_(getOptionalParam<int, double>(
              "maxDz", config, [&scales](double value) { return scales.to_hw_z0_ceil(value); })),
          minDRSquared_(getOptionalParam<int, double>(
              "minDR", config, [&scales](double value) { return scales.to_hw_dRSquared_floor(value); })),
          maxDRSquared_(getOptionalParam<int, double>(
              "maxDR", config, [&scales](double value) { return scales.to_hw_dRSquared_ceil(value); })),
          minInvMassSqrDiv2_scale1_(getOptionalParam<int64_t, double>(
              "minInvMass",
              config,
              [&](double value) {
                return std::floor(scales.to_hw_InvMassSqrDiv2(value) * coshEtaLUT_.output_scale());
              })),
          maxInvMassSqrDiv2_scale1_(getOptionalParam<int64_t, double>(
              "maxInvMass",
              config,
              [&](double value) {
                return std::ceil(scales.to_hw_InvMassSqrDiv2(value) * coshEtaLUT_.output_scale());
              })),
          minInvMassSqrDiv2_scale2_(getOptionalParam<int64_t, double>(
              "minInvMass",
              config,
              [&](double value) {
                return std::floor(scales.to_hw_InvMassSqrDiv2(value) * coshEtaLUT2_.output_scale());
              })),
          maxInvMassSqrDiv2_scale2_(getOptionalParam<int64_t, double>(
              "maxInvMass",
              config,
              [&](double value) {
                return std::ceil(scales.to_hw_InvMassSqrDiv2(value) * coshEtaLUT2_.output_scale());
              })),
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
          minPTSquared_(getOptionalParam<int64_t, double>(
              "minCombPt",
              config,
              [&](double value) { return std::floor(scales.to_hw_PtSquared(value) * cosPhiLUT_.output_scale()); })),
          maxPTSquared_(getOptionalParam<int64_t, double>(
              "maxCombPt",
              config,
              [&](double value) { return std::ceil(scales.to_hw_PtSquared(value) * cosPhiLUT_.output_scale()); })),
          minInvMassSqrOver2DRSqr_scale1_(getOptionalParam<double, double>(
              "minInvMassOverDR",
              config,
              [&](double value) {
                return std::floor(scales.to_hw_InvMassSqrOver2DR(value) * coshEtaLUT_.output_scale());
              })),
          maxInvMassSqrOver2DRSqr_scale1_(getOptionalParam<double, double>(
              "maxInvMassOverDR",
              config,
              [&](double value) {
                return std::ceil(scales.to_hw_InvMassSqrOver2DR(value) * coshEtaLUT_.output_scale());
              })),
          minInvMassSqrOver2DRSqr_scale2_(getOptionalParam<double, double>(
              "minInvMassOverDR",
              config,
              [&](double value) {
                return std::floor(scales.to_hw_InvMassSqrOver2DR(value) * coshEtaLUT2_.output_scale());
              })),
          maxInvMassSqrOver2DRSqr_scale2_(getOptionalParam<double, double>(
              "maxInvMassOverDR",
              config,
              [&](double value) {
                return std::ceil(scales.to_hw_InvMassSqrOver2DR(value) * coshEtaLUT2_.output_scale());
              })),
          os_(config.getParameter<bool>("os")),
          ss_(config.getParameter<bool>("ss")),
          enable_sanity_checks_(enable_sanity_checks),
          inv_mass_checks_(inv_mass_checks) {}

    bool checkObjects(const P2GTCandidate& obj1,
                      const P2GTCandidate& obj2,
                      InvariantMassErrorCollection& massErrors) const {
      bool res = true;

      std::optional<uint32_t> dEta;

      if (minDEta_ || maxDEta_ || minDRSquared_ || maxDRSquared_ || minInvMassSqrDiv2_scale1_ ||
          maxInvMassSqrDiv2_scale1_ || minInvMassSqrDiv2_scale2_ || maxInvMassSqrDiv2_scale2_ ||
          minInvMassSqrOver2DRSqr_scale1_ || maxInvMassSqrOver2DRSqr_scale1_ || minInvMassSqrOver2DRSqr_scale2_ ||
          maxInvMassSqrOver2DRSqr_scale2_) {
        dEta = std::abs(obj1.hwEta().to_int() - obj2.hwEta().to_int());
        res &= minDEta_ ? dEta > minDEta_ : true;
        res &= maxDEta_ ? dEta < maxDEta_ : true;
      }

      constexpr int HW_PI = 1 << (P2GTCandidate::hwPhi_t::width - 1);  // assumes phi in [-pi, pi)

      std::optional<uint32_t> dPhi;

      if (minDPhi_ || maxDPhi_ || minDRSquared_ || maxDRSquared_ || minInvMassSqrDiv2_scale1_ ||
          maxInvMassSqrDiv2_scale1_ || minInvMassSqrDiv2_scale2_ || maxInvMassSqrDiv2_scale2_ || minTransMassSqrDiv2_ ||
          maxTransMassSqrDiv2_ || minPTSquared_ || maxPTSquared_ || minInvMassSqrOver2DRSqr_scale1_ ||
          maxInvMassSqrOver2DRSqr_scale1_ || minInvMassSqrOver2DRSqr_scale2_ || maxInvMassSqrOver2DRSqr_scale2_) {
        // Ensure dPhi is always the smaller angle, i.e. always between [0, pi]
        dPhi = HW_PI - std::abs(std::abs(obj1.hwPhi().to_int() - obj2.hwPhi().to_int()) - HW_PI);
      }

      res &= minDPhi_ ? dPhi > minDPhi_ : true;
      res &= maxDPhi_ ? dPhi < maxDPhi_ : true;

      if (minDz_ || maxDz_) {
        uint32_t dZ = abs(obj1.hwZ0() - obj2.hwZ0());
        res &= minDz_ ? dZ > minDz_ : true;
        res &= maxDz_ ? dZ < maxDz_ : true;
      }

      uint32_t dRSquared = 0;
      if (minDRSquared_ || maxDRSquared_ || minInvMassSqrOver2DRSqr_scale1_ || maxInvMassSqrOver2DRSqr_scale1_ ||
          minInvMassSqrOver2DRSqr_scale2_ || maxInvMassSqrOver2DRSqr_scale2_) {
        dRSquared = dEta.value() * dEta.value() + dPhi.value() * dPhi.value();
        res &= minDRSquared_ ? dRSquared > minDRSquared_ : true;
        res &= maxDRSquared_ ? dRSquared < maxDRSquared_ : true;
      }

      res &= os_ ? obj1.hwCharge() != obj2.hwCharge() : true;
      res &= ss_ ? obj1.hwCharge() == obj2.hwCharge() : true;

      int32_t lutCoshDEta = 0;
      if (dEta) {
        lutCoshDEta = dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT
                          ? coshEtaLUT_[dEta.value()]
                          : coshEtaLUT2_[dEta.value() - L1GTSingleInOutLUT::DETA_LUT_SPLIT];
      }

      // Optimization: (cos(x + pi) = -cos(x), x in [0, pi))
      int32_t lutCosDPhi = 0;
      if (dPhi) {
        lutCosDPhi = dPhi >= HW_PI ? -cosPhiLUT_[dPhi.value()] : cosPhiLUT_[dPhi.value()];
      }

      if (enable_sanity_checks_ && dEta && dPhi) {
        // Check whether the LUT error is smaller or equal than the expected maximum LUT error
        double coshEtaLUTMax =
            dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT ? coshEtaLUT_.hwMax_error() : coshEtaLUT2_.hwMax_error();
        double etaLUTScale =
            dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT ? coshEtaLUT_.output_scale() : coshEtaLUT2_.output_scale();

        if (std::abs(lutCoshDEta - etaLUTScale * std::cosh(dEta.value() * scales_.eta_lsb())) > coshEtaLUTMax) {
          edm::LogError("COSH LUT") << "Difference larger than max LUT error: " << coshEtaLUTMax
                                    << ", lut: " << lutCoshDEta
                                    << ", calc: " << etaLUTScale * std::cosh(dEta.value() * scales_.eta_lsb())
                                    << ", dEta: " << dEta.value() << ", scale: " << etaLUTScale;
        }

        if (std::abs(lutCosDPhi - cosPhiLUT_.output_scale() * std::cos(dPhi.value() * scales_.phi_lsb())) >
            cosPhiLUT_.hwMax_error()) {
          edm::LogError("COS LUT") << "Difference larger than max LUT error: " << cosPhiLUT_.hwMax_error()
                                   << ", lut: " << lutCosDPhi << ", calc: "
                                   << cosPhiLUT_.output_scale() * std::cos(dPhi.value() * scales_.phi_lsb());
        }
      }

      int64_t invMassSqrDiv2 = 0;
      if (minInvMassSqrDiv2_scale1_ || maxInvMassSqrDiv2_scale1_ || minInvMassSqrDiv2_scale2_ ||
          maxInvMassSqrDiv2_scale2_ || minInvMassSqrOver2DRSqr_scale1_ || maxInvMassSqrOver2DRSqr_scale1_ ||
          minInvMassSqrOver2DRSqr_scale2_ || maxInvMassSqrOver2DRSqr_scale2_) {
        if (dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT) {
          // dEta [0, 2pi)
          invMassSqrDiv2 = obj1.hwPT().to_int64() * obj2.hwPT().to_int64() * (lutCoshDEta - lutCosDPhi);
          res &= minInvMassSqrDiv2_scale1_ ? invMassSqrDiv2 > minInvMassSqrDiv2_scale1_ : true;
          res &= maxInvMassSqrDiv2_scale1_ ? invMassSqrDiv2 < maxInvMassSqrDiv2_scale1_ : true;
        } else {
          // dEta [2pi, 4pi), ignore cos
          invMassSqrDiv2 = obj1.hwPT().to_int64() * obj2.hwPT().to_int64() * lutCoshDEta;
          res &= minInvMassSqrDiv2_scale2_ ? invMassSqrDiv2 > minInvMassSqrDiv2_scale2_ : true;
          res &= maxInvMassSqrDiv2_scale2_ ? invMassSqrDiv2 < maxInvMassSqrDiv2_scale2_ : true;
        }

        if (inv_mass_checks_) {
          double precInvMass =
              scales_.pT_lsb() *
              std::sqrt(2 * obj1.hwPT().to_double() * obj2.hwPT().to_double() *
                        (std::cosh(dEta.value() * scales_.eta_lsb()) - std::cos(dPhi.value() * scales_.phi_lsb())));

          double lutInvMass =
              scales_.pT_lsb() * std::sqrt(2 * static_cast<double>(invMassSqrDiv2) /
                                           (dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT ? coshEtaLUT_.output_scale()
                                                                                      : coshEtaLUT2_.output_scale()));

          double error = std::abs(precInvMass - lutInvMass);
          massErrors.emplace_back(InvariantMassError{error, error / precInvMass, precInvMass});
        }
      }

      if (minPTSquared_ || maxPTSquared_) {
        int64_t pTSquared = obj1.hwPT().to_int64() * obj1.hwPT().to_int64() *
                                static_cast<int64_t>(std::round(cosPhiLUT_.output_scale())) +
                            obj2.hwPT().to_int64() * obj2.hwPT().to_int64() *
                                static_cast<int64_t>(std::round(cosPhiLUT_.output_scale())) +
                            2 * obj1.hwPT().to_int64() * obj2.hwPT().to_int64() * lutCosDPhi;
        res &= minPTSquared_ ? pTSquared > minPTSquared_.value() : true;
        res &= maxPTSquared_ ? pTSquared < maxPTSquared_.value() : true;
      }

      if (minTransMassSqrDiv2_ || maxTransMassSqrDiv2_) {
        int64_t transMassDiv2 = obj1.hwPT().to_int64() * obj2.hwPT().to_int64() *
                                (static_cast<int64_t>(std::round(cosPhiLUT_.output_scale())) - lutCosDPhi);
        res &= minTransMassSqrDiv2_ ? transMassDiv2 > minTransMassSqrDiv2_.value() : true;
        res &= maxTransMassSqrDiv2_ ? transMassDiv2 < maxTransMassSqrDiv2_.value() : true;
      }

      if (minInvMassSqrOver2DRSqr_scale1_ || maxInvMassSqrOver2DRSqr_scale1_ || minInvMassSqrOver2DRSqr_scale2_ ||
          maxInvMassSqrOver2DRSqr_scale2_) {
        ap_uint<96> invMassSqrDiv2Shift = ap_uint<96>(invMassSqrDiv2)
                                          << L1GTScales::INV_MASS_SQR_OVER_2_DR_SQR_RESOLUTION;

        if (dEta < L1GTSingleInOutLUT::DETA_LUT_SPLIT) {
          res &= minInvMassSqrOver2DRSqr_scale1_
                     ? invMassSqrDiv2Shift > minInvMassSqrOver2DRSqr_scale1_.value() * dRSquared
                     : true;
          res &= maxInvMassSqrOver2DRSqr_scale1_
                     ? invMassSqrDiv2Shift < maxInvMassSqrOver2DRSqr_scale1_.value() * dRSquared
                     : true;
        } else {
          res &= minInvMassSqrOver2DRSqr_scale2_
                     ? invMassSqrDiv2Shift > minInvMassSqrOver2DRSqr_scale2_.value() * dRSquared
                     : true;
          res &= maxInvMassSqrOver2DRSqr_scale2_
                     ? invMassSqrDiv2Shift < maxInvMassSqrOver2DRSqr_scale2_.value() * dRSquared
                     : true;
        }
      }

      return res;
    }

    static void fillLUTDescriptions(edm::ParameterSetDescription& desc) {
      edm::ParameterSetDescription coshLUTDesc;
      L1GTSingleInOutLUT::fillLUTDescriptions(coshLUTDesc);
      desc.add<edm::ParameterSetDescription>("cosh_eta_lut", coshLUTDesc);

      edm::ParameterSetDescription coshLUT2Desc;
      L1GTSingleInOutLUT::fillLUTDescriptions(coshLUT2Desc);
      desc.add<edm::ParameterSetDescription>("cosh_eta_lut2", coshLUT2Desc);

      edm::ParameterSetDescription cosLUTDesc;
      L1GTSingleInOutLUT::fillLUTDescriptions(cosLUTDesc);
      desc.add<edm::ParameterSetDescription>("cos_phi_lut", cosLUTDesc);
    }

    static void fillPSetDescription(edm::ParameterSetDescription& desc) {
      desc.addOptional<double>("minDEta");
      desc.addOptional<double>("maxDEta");
      desc.addOptional<double>("minDPhi");
      desc.addOptional<double>("maxDPhi");
      desc.addOptional<double>("minDR");
      desc.addOptional<double>("maxDR");
      desc.addOptional<double>("minDz");
      desc.addOptional<double>("maxDz");
      desc.addOptional<double>("minInvMass");
      desc.addOptional<double>("maxInvMass");
      desc.addOptional<double>("minTransMass");
      desc.addOptional<double>("maxTransMass");
      desc.addOptional<double>("minCombPt");
      desc.addOptional<double>("maxCombPt");
      desc.addOptional<double>("minInvMassOverDR");
      desc.addOptional<double>("maxInvMassOverDR");
      desc.add<bool>("os", false);
      desc.add<bool>("ss", false);
    }

  private:
    const L1GTScales& scales_;

    const L1GTSingleInOutLUT coshEtaLUT_;   // [0, 2pi)
    const L1GTSingleInOutLUT coshEtaLUT2_;  // [2pi, 4pi)
    const L1GTSingleInOutLUT cosPhiLUT_;

    const std::optional<int> minDEta_;
    const std::optional<int> maxDEta_;
    const std::optional<int> minDPhi_;
    const std::optional<int> maxDPhi_;
    const std::optional<int> minDz_;
    const std::optional<int> maxDz_;

    const std::optional<int> minDRSquared_;
    const std::optional<int> maxDRSquared_;

    const std::optional<int64_t> minInvMassSqrDiv2_scale1_;
    const std::optional<int64_t> maxInvMassSqrDiv2_scale1_;

    const std::optional<int64_t> minInvMassSqrDiv2_scale2_;
    const std::optional<int64_t> maxInvMassSqrDiv2_scale2_;

    const std::optional<int64_t> minTransMassSqrDiv2_;
    const std::optional<int64_t> maxTransMassSqrDiv2_;

    const std::optional<int64_t> minPTSquared_;
    const std::optional<int64_t> maxPTSquared_;

    const std::optional<int64_t> minInvMassSqrOver2DRSqr_scale1_;
    const std::optional<int64_t> maxInvMassSqrOver2DRSqr_scale1_;

    const std::optional<int64_t> minInvMassSqrOver2DRSqr_scale2_;
    const std::optional<int64_t> maxInvMassSqrOver2DRSqr_scale2_;

    const bool os_;  // Opposite sign
    const bool ss_;  // Same sign

    const bool enable_sanity_checks_;
    const bool inv_mass_checks_;
  };

}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTCorrelationalCut_h
