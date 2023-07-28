#ifndef L1Trigger_Phase2L1GT_L1GTDeltaCut_h
#define L1Trigger_Phase2L1GT_L1GTDeltaCut_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTInvariantMassError.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"

#include "L1GTSingleInOutLUT.h"
#include "L1GTOptionalParam.h"

#include <optional>

namespace l1t {

  class L1GTDeltaCut {
  public:
    static constexpr uint32_t DETA_LUT_SPLIT = 1 << 13;  // hw 2pi

    L1GTDeltaCut(const edm::ParameterSet& config,
                 const edm::ParameterSet& lutConfig,
                 const L1GTScales& scales,
                 bool enable_sanity_checks = false,
                 bool inv_mass_checks = false)
        : scales_(scales),
          coshEtaLUT_(lutConfig.getParameterSet("cosh_eta_lut")),
          coshEtaLUT2_(lutConfig.getParameterSet("cosh_eta_lut2")),
          cosPhiLUT_(lutConfig.getParameterSet("cos_phi_lut")),
          minDEta_(getOptionalParam<int, double>(
              "minDEta", config, [&scales](double value) { return scales.to_hw_eta(value); })),
          maxDEta_(getOptionalParam<int, double>(
              "maxDEta", config, [&scales](double value) { return scales.to_hw_eta(value); })),
          minDPhi_(getOptionalParam<int, double>(
              "minDPhi", config, [&scales](double value) { return scales.to_hw_phi(value); })),
          maxDPhi_(getOptionalParam<int, double>(
              "maxDPhi", config, [&scales](double value) { return scales.to_hw_phi(value); })),
          minDz_(getOptionalParam<int, double>(
              "minDz", config, [&scales](double value) { return scales.to_hw_z0(value); })),
          maxDz_(getOptionalParam<int, double>(
              "maxDz", config, [&scales](double value) { return scales.to_hw_z0(value); })),
          minDRSquared_(getOptionalParam<int, double>(
              "minDR", config, [&scales](double value) { return scales.to_hw_dRSquared(value); })),
          maxDRSquared_(getOptionalParam<int, double>(
              "maxDR", config, [&scales](double value) { return scales.to_hw_dRSquared(value); })),
          minInvMassSqrDiv2_(getOptionalParam<double, double>(
              "minInvMass", config, [&scales](double value) { return scales.to_hw_InvMassSqrDiv2(value); })),
          maxInvMassSqrDiv2_(getOptionalParam<double, double>(
              "maxInvMass", config, [&scales](double value) { return scales.to_hw_InvMassSqrDiv2(value); })),
          minTransMassSqrDiv2_(getOptionalParam<double, double>(
              "minTransMass", config, [&scales](double value) { return scales.to_hw_TransMassSqrDiv2(value); })),
          maxTransMassSqrDiv2_(getOptionalParam<double, double>(
              "maxTransMass", config, [&scales](double value) { return scales.to_hw_TransMassSqrDiv2(value); })),
          minPTSquared_(getOptionalParam<double, double>(
              "minCombPt", config, [&scales](double value) { return scales.to_hw_PtSquared(value); })),
          maxPTSquared_(getOptionalParam<double, double>(
              "maxCombPt", config, [&scales](double value) { return scales.to_hw_PtSquared(value); })),
          os_(config.getParameter<bool>("os")),
          ss_(config.getParameter<bool>("ss")),
          enable_sanity_checks_(enable_sanity_checks),
          inv_mass_checks_(inv_mass_checks) {}

    bool checkObjects(const P2GTCandidate& obj1,
                      const P2GTCandidate& obj2,
                      InvariantMassErrorCollection& massErrors) const {
      bool res = true;

      std::optional<uint32_t> dEta;

      if (minDEta_ || maxDEta_ || minDRSquared_ || maxDRSquared_ || minInvMassSqrDiv2_ || maxInvMassSqrDiv2_) {
        dEta = (obj1.hwEta() > obj2.hwEta()) ? obj1.hwEta().to_int() - obj2.hwEta().to_int()
                                             : obj2.hwEta().to_int() - obj1.hwEta().to_int();
        res &= minDEta_ ? dEta > minDEta_ : true;
        res &= maxDEta_ ? dEta < maxDEta_ : true;
      }

      constexpr int HW_PI = 1 << (P2GTCandidate::hwPhi_t::width - 1);  // assumes phi in [-pi, pi)

      // Ensure dPhi is always the smaller angle, i.e. always between [0, pi]
      std::optional<uint32_t> dPhi;

      if (minDPhi_ || maxDPhi_ || minDRSquared_ || maxDRSquared_ || minInvMassSqrDiv2_ || maxInvMassSqrDiv2_ ||
          minTransMassSqrDiv2_ || maxTransMassSqrDiv2_ || minPTSquared_ || maxPTSquared_) {
        dPhi = HW_PI - abs(abs(obj1.hwPhi().to_int() - obj2.hwPhi().to_int()) - HW_PI);
      }

      res &= minDPhi_ ? dPhi > minDPhi_ : true;
      res &= maxDPhi_ ? dPhi < maxDPhi_ : true;

      if (minDz_ || maxDz_) {
        uint32_t dZ = abs(obj1.hwZ0() - obj2.hwZ0());
        res &= minDz_ ? dZ > minDz_ : true;
        res &= maxDz_ ? dZ < maxDz_ : true;
      }

      if (minDRSquared_ || maxDRSquared_) {
        uint32_t dRSquared = dEta.value() * dEta.value() + dPhi.value() * dPhi.value();
        res &= minDRSquared_ ? dRSquared > minDRSquared_ : true;
        res &= maxDRSquared_ ? dRSquared < maxDRSquared_ : true;
      }

      res &= os_ ? obj1.hwCharge() != obj2.hwCharge() : true;
      res &= ss_ ? obj1.hwCharge() == obj2.hwCharge() : true;

      int32_t lutCoshDEta = 0;
      if (dEta) {
        lutCoshDEta = dEta < DETA_LUT_SPLIT ? coshEtaLUT_[dEta.value()] : coshEtaLUT2_[dEta.value() - DETA_LUT_SPLIT];
      }

      // Optimization: (cos(x + pi) = -cos(x), x in [0, pi))
      int32_t lutCosDPhi = 0;
      if (dPhi) {
        lutCosDPhi = dPhi >= HW_PI ? -cosPhiLUT_[dPhi.value()] : cosPhiLUT_[dPhi.value()];
      }

      if (enable_sanity_checks_ && dEta && dPhi) {
        // Check whether the LUT error is smaller or equal than the expected maximum LUT error
        double coshEtaLUTMax = dEta < DETA_LUT_SPLIT ? coshEtaLUT_.hwMax_error() : coshEtaLUT2_.hwMax_error();
        double etaLUTScale = dEta < DETA_LUT_SPLIT ? coshEtaLUT_.output_scale() : coshEtaLUT2_.output_scale();

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

      if (minInvMassSqrDiv2_ || maxInvMassSqrDiv2_) {
        int64_t invMassSqrDiv2;
        if (dEta < DETA_LUT_SPLIT) {
          // dEta [0, 2pi)
          invMassSqrDiv2 = obj1.hwPT().to_int64() * obj2.hwPT().to_int64() * (lutCoshDEta - lutCosDPhi);
          res &= minInvMassSqrDiv2_
                     ? invMassSqrDiv2 > std::round(minInvMassSqrDiv2_.value() * coshEtaLUT_.output_scale())
                     : true;
          res &= maxInvMassSqrDiv2_
                     ? invMassSqrDiv2 < std::round(maxInvMassSqrDiv2_.value() * coshEtaLUT_.output_scale())
                     : true;
        } else {
          // dEta [2pi, 4pi), ignore cos
          invMassSqrDiv2 = obj1.hwPT().to_int64() * obj2.hwPT().to_int64() * lutCoshDEta;
          res &= minInvMassSqrDiv2_
                     ? invMassSqrDiv2 > std::round(minInvMassSqrDiv2_.value() * coshEtaLUT2_.output_scale())
                     : true;
          res &= maxInvMassSqrDiv2_
                     ? invMassSqrDiv2 < std::round(maxInvMassSqrDiv2_.value() * coshEtaLUT2_.output_scale())
                     : true;
        }

        if (inv_mass_checks_) {
          double precInvMass =
              scales_.pT_lsb() *
              std::sqrt(2 * obj1.hwPT().to_double() * obj2.hwPT().to_double() *
                        (std::cosh(dEta.value() * scales_.eta_lsb()) - std::cos(dPhi.value() * scales_.phi_lsb())));

          double lutInvMass = scales_.pT_lsb() * std::sqrt(2 * static_cast<double>(invMassSqrDiv2) /
                                                           (dEta < DETA_LUT_SPLIT ? coshEtaLUT_.output_scale()
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
        res &= minPTSquared_ ? pTSquared > std::round(minPTSquared_.value() * cosPhiLUT_.output_scale()) : true;
        res &= maxPTSquared_ ? pTSquared < std::round(maxPTSquared_.value() * cosPhiLUT_.output_scale()) : true;
      }

      if (minTransMassSqrDiv2_ || maxTransMassSqrDiv2_) {
        int64_t transMassDiv2 = obj1.hwPT().to_int64() * obj2.hwPT().to_int64() *
                                (static_cast<int64_t>(coshEtaLUT_.output_scale()) - lutCosDPhi);
        res &= minTransMassSqrDiv2_
                   ? transMassDiv2 > std::round(minTransMassSqrDiv2_.value() * coshEtaLUT_.output_scale())
                   : true;
        res &= maxTransMassSqrDiv2_
                   ? transMassDiv2 < std::round(maxTransMassSqrDiv2_.value() * coshEtaLUT_.output_scale())
                   : true;
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

    const std::optional<double> minInvMassSqrDiv2_;
    const std::optional<double> maxInvMassSqrDiv2_;
    const std::optional<double> minTransMassSqrDiv2_;
    const std::optional<double> maxTransMassSqrDiv2_;

    const std::optional<double> minPTSquared_;
    const std::optional<double> maxPTSquared_;

    const bool os_;  // Opposite sign
    const bool ss_;  // Same sign

    const bool enable_sanity_checks_;
    const bool inv_mass_checks_;
  };

}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTDeltaCut_h
