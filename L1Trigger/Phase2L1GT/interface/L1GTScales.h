#ifndef L1Trigger_Phase2L1GT_L1GTScales_h
#define L1Trigger_Phase2L1GT_L1GTScales_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace l1t {
  class L1GTScales {
  public:
    static constexpr int RELATIVE_ISOLATION_RESOLUTION = 18;  // Resolution = 1/2^RELATIVE_ISOLATION_RESOLUTION
    /* INV_MASS_SQR_OVER_2_DR_SQR_RESOLUTION originates from a simple analysis that yielded that the smallest
       delta in hardware values of M^2/(2 dR^2) = 2.93326e-06 => log2(2.93326e-06) = -18.38 */
    static constexpr int INV_MASS_SQR_OVER_2_DR_SQR_RESOLUTION = 19;

    L1GTScales(double pT_lsb,
               double phi_lsb,
               double eta_lsb,
               double z0_lsb,
               //double dD_lsb,
               double isolationPT_lsb,
               double beta_lsb,
               double mass_lsb,
               double seed_pT_lsb,
               double seed_dZ_lsb,
               double scalarSumPT_lsb,
               double sum_pT_pv_lsb,
               int pos_chg,
               int neg_chg);

    L1GTScales(const edm::ParameterSet &);

    static void fillPSetDescription(edm::ParameterSetDescription &);

    int to_hw_pT_ceil(double value) const { return std::ceil(value / pT_lsb_); };
    int to_hw_phi_ceil(double value) const { return std::ceil(value / phi_lsb_); };
    int to_hw_eta_ceil(double value) const { return std::ceil(value / eta_lsb_); };
    int to_hw_z0_ceil(double value) const { return std::ceil(value / z0_lsb_); };
    // int to_hw_d0(double value) const { return std::ceil(value / d0_lsb_); };
    int to_hw_relative_isolationPT_ceil(double value) const {
      return std::ceil(pT_lsb_ * value * std::pow(2, RELATIVE_ISOLATION_RESOLUTION) / isolationPT_lsb_);
    }
    int to_hw_isolationPT_ceil(double value) const { return std::ceil(value / isolationPT_lsb_); }
    int to_hw_beta_ceil(double value) const { return std::ceil(value / beta_lsb_); };
    int to_hw_mass_ceil(double value) const { return std::ceil(value / mass_lsb_); };
    int to_hw_seed_pT_ceil(double value) const { return std::ceil(value / seed_pT_lsb_); };
    int to_hw_seed_z0_ceil(double value) const { return std::ceil(value / seed_z0_lsb_); };
    int to_hw_scalarSumPT_ceil(double value) const { return std::ceil(value / scalarSumPT_lsb_); };
    int to_hw_sum_pT_pv_ceil(double value) const { return std::ceil(value / sum_pT_pv_lsb_); };

    int to_hw_dRSquared_ceil(double value) const { return std::ceil(value * value / (eta_lsb_ * eta_lsb_)); }

    double to_hw_InvMassSqrDiv2(double value) const { return value * value / (2 * pT_lsb_ * pT_lsb_); }
    double to_hw_TransMassSqrDiv2(double value) const { return value * value / (2 * pT_lsb_ * pT_lsb_); }

    double to_hw_PtSquared(double value) const { return value * value / (pT_lsb_ * pT_lsb_); }
    double to_hw_InvMassSqrOver2DR(double value) const {
      return value * value * eta_lsb_ * eta_lsb_ * std::pow(2, INV_MASS_SQR_OVER_2_DR_SQR_RESOLUTION) /
             (2 * pT_lsb_ * pT_lsb_);
    }

    int to_hw_pT_floor(double value) const { return std::floor(value / pT_lsb_); };
    int to_hw_phi_floor(double value) const { return std::floor(value / phi_lsb_); };
    int to_hw_eta_floor(double value) const { return std::floor(value / eta_lsb_); };
    int to_hw_z0_floor(double value) const { return std::floor(value / z0_lsb_); };
    // int to_hw_d0(double value) const { return std::floor(value / d0_lsb_); };
    int to_hw_relative_isolationPT_floor(double value) const {
      return std::floor(pT_lsb_ * value * std::pow(2, RELATIVE_ISOLATION_RESOLUTION) / isolationPT_lsb_);
    }
    int to_hw_isolationPT_floor(double value) const { return std::floor(value / isolationPT_lsb_); }
    int to_hw_beta_floor(double value) const { return std::floor(value / beta_lsb_); };
    int to_hw_mass_floor(double value) const { return std::floor(value / mass_lsb_); };
    int to_hw_seed_pT_floor(double value) const { return std::floor(value / seed_pT_lsb_); };
    int to_hw_seed_z0_floor(double value) const { return std::floor(value / seed_z0_lsb_); };
    int to_hw_scalarSumPT_floor(double value) const { return std::floor(value / scalarSumPT_lsb_); };
    int to_hw_sum_pT_pv_floor(double value) const { return std::floor(value / sum_pT_pv_lsb_); };

    int to_hw_dRSquared_floor(double value) const { return std::floor(value * value / (eta_lsb_ * eta_lsb_)); }

    double to_pT(int value) const { return value * pT_lsb_; };
    double to_phi(int value) const { return value * phi_lsb_; };
    double to_eta(int value) const { return value * eta_lsb_; };
    double to_z0(int value) const { return value * z0_lsb_; };
    double to_isolationPT(int value) const { return value * isolationPT_lsb_; }
    double to_scalarSumPT(int value) const { return value * scalarSumPT_lsb_; };
    int to_chg(int value) const { return value == pos_chg_ ? +1 : value == neg_chg_ ? -1 : 0; }

    double pT_lsb() const { return pT_lsb_; }
    double phi_lsb() const { return phi_lsb_; }
    double eta_lsb() const { return eta_lsb_; }
    double z0_lsb() const { return z0_lsb_; }
    double isolationPT_lsb() const { return isolationPT_lsb_; }
    //const double dD_lsb_;
    double beta_lsb() const { return beta_lsb_; }
    double mass_lsb() const { return mass_lsb_; }
    double seed_pT_lsb() const { return seed_pT_lsb_; }
    double seed_z0_lsb() const { return seed_z0_lsb_; }
    double scalarSumPT_lsb() const { return scalarSumPT_lsb_; }
    double sum_pT_pv_lsb() const { return sum_pT_pv_lsb_; }
    int pos_chg() const { return pos_chg_; }
    int neg_chg() const { return neg_chg_; }

  private:
    const double pT_lsb_;
    const double phi_lsb_;
    const double eta_lsb_;
    const double z0_lsb_;
    //const double dD_lsb_;
    const double isolationPT_lsb_;
    const double beta_lsb_;
    const double mass_lsb_;
    const double seed_pT_lsb_;
    const double seed_z0_lsb_;
    const double scalarSumPT_lsb_;
    const double sum_pT_pv_lsb_;
    const int pos_chg_;
    const int neg_chg_;
  };
}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTScales_h
