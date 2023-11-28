#ifndef L1Trigger_Phase2L1GT_L1GTScales_h
#define L1Trigger_Phase2L1GT_L1GTScales_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace l1t {
  class L1GTScales {
    static constexpr int RELATIVE_ISOLATION_RESOLUTION = 10;  // Resolution = 1/2^RELATIVE_ISOLATION_RESOLUTION

  public:
    L1GTScales(double pT_lsb,
               double phi_lsb,
               double eta_lsb,
               double z0_lsb,
               //double dD_lsb,
               double isolation_lsb,
               double beta_lsb,
               double mass_lsb,
               double seed_pT_lsb,
               double seed_dZ_lsb,
               double sca_sum_lsb,
               double sum_pT_pv_lsb,
               int pos_chg,
               int neg_chg);

    L1GTScales(const edm::ParameterSet &);

    static void fillPSetDescription(edm::ParameterSetDescription &);

    int to_hw_pT(double value) const { return std::round(value / pT_lsb_); };
    int to_hw_phi(double value) const { return std::round(value / phi_lsb_); };
    int to_hw_eta(double value) const { return std::round(value / eta_lsb_); };
    int to_hw_z0(double value) const { return std::round(value / z0_lsb_); };
    // int to_hw_d0(double value) const { return std::round(value / d0_lsb_); };
    int to_hw_isolation(double value) const {
      return std::round(pT_lsb_ * value * std::pow(2, isolation_shift_) / isolation_lsb_);
    }
    int to_hw_beta(double value) const { return std::round(value / beta_lsb_); };
    int to_hw_mass(double value) const { return std::round(value / mass_lsb_); };
    int to_hw_seed_pT(double value) const { return std::round(value / seed_pT_lsb_); };
    int to_hw_seed_z0(double value) const { return std::round(value / seed_z0_lsb_); };
    int to_hw_sca_sum(double value) const { return std::round(value / sca_sum_lsb_); };
    int to_hw_sum_pT_pv(double value) const { return std::round(value / sum_pT_pv_lsb_); };

    int to_hw_dRSquared(double value) const { return std::round(value * value / (eta_lsb_ * eta_lsb_)); }

    double to_hw_InvMassSqrDiv2(double value) const { return value * value / (2 * pT_lsb_ * pT_lsb_); }
    double to_hw_TransMassSqrDiv2(double value) const { return value * value / (2 * pT_lsb_ * pT_lsb_); }

    double to_hw_PtSquared(double value) const { return value * value / (pT_lsb_ * pT_lsb_); }

    double to_pT(int value) const { return value * pT_lsb_; };
    double to_phi(int value) const { return value * phi_lsb_; };
    double to_eta(int value) const { return value * eta_lsb_; };
    double to_z0(int value) const { return value * z0_lsb_; };
    double to_sca_sum(int value) const { return value * sca_sum_lsb_; };
    int to_chg(int value) const { return value == pos_chg_ ? +1 : value == neg_chg_ ? -1 : 0; }

    double pT_lsb() const { return pT_lsb_; }
    double phi_lsb() const { return phi_lsb_; }
    double eta_lsb() const { return eta_lsb_; }
    double z0_lsb() const { return z0_lsb_; }
    double isolation_lsb() const { return isolation_lsb_; }
    //const double dD_lsb_;
    double beta_lsb() const { return beta_lsb_; }
    double mass_lsb() const { return mass_lsb_; }
    double seed_pT_lsb() const { return seed_pT_lsb_; }
    double seed_z0_lsb() const { return seed_z0_lsb_; }
    double sca_sum_lsb() const { return sca_sum_lsb_; }
    double sum_pT_pv_lsb() const { return sum_pT_pv_lsb_; }
    int pos_chg() const { return pos_chg_; }
    int neg_chg() const { return neg_chg_; }
    int isolation_shift() const { return isolation_shift_; }

  private:
    const double pT_lsb_;
    const double phi_lsb_;
    const double eta_lsb_;
    const double z0_lsb_;
    //const double dD_lsb_;
    const double isolation_lsb_;
    const double isolation_shift_;
    const double beta_lsb_;
    const double mass_lsb_;
    const double seed_pT_lsb_;
    const double seed_z0_lsb_;
    const double sca_sum_lsb_;
    const double sum_pT_pv_lsb_;
    const int pos_chg_;
    const int neg_chg_;
  };
}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTScales_h
