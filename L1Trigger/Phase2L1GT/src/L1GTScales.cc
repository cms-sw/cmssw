#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace l1t {
  L1GTScales::L1GTScales(double pT_lsb,
                         double phi_lsb,
                         double eta_lsb,
                         double z0_lsb,
                         //double d0_lsb,
                         double isolation_lsb,
                         double beta_lsb,
                         double mass_lsb,
                         double seed_pT_lsb,
                         double seed_z0_lsb,
                         double sca_sum_lsb,
                         double sum_pT_pv_lsb,
                         int pos_chg,
                         int neg_chg)
      : pT_lsb_(pT_lsb),
        phi_lsb_(phi_lsb),
        eta_lsb_(eta_lsb),
        z0_lsb_(z0_lsb),
        //d0_lsb_(d0_lsb),
        isolation_lsb_(isolation_lsb),
        isolation_shift_(RELATIVE_ISOLATION_RESOLUTION + std::log2(isolation_lsb_ / pT_lsb_)),
        beta_lsb_(beta_lsb),
        mass_lsb_(mass_lsb),
        seed_pT_lsb_(seed_pT_lsb),
        seed_z0_lsb_(seed_z0_lsb),
        sca_sum_lsb_(sca_sum_lsb),
        sum_pT_pv_lsb_(sum_pT_pv_lsb),
        pos_chg_(pos_chg),
        neg_chg_(neg_chg) {}

  L1GTScales::L1GTScales(const edm::ParameterSet& config)
      : pT_lsb_(config.getParameter<double>("pT_lsb")),
        phi_lsb_(config.getParameter<double>("phi_lsb")),
        eta_lsb_(config.getParameter<double>("eta_lsb")),
        z0_lsb_(config.getParameter<double>("z0_lsb")),
        //d0_lsb_(config.getParameter<double>("d0_lsb")),
        isolation_lsb_(config.getParameter<double>("isolation_lsb")),
        isolation_shift_(RELATIVE_ISOLATION_RESOLUTION + std::log2(isolation_lsb_ / pT_lsb_)),
        beta_lsb_(config.getParameter<double>("beta_lsb")),
        mass_lsb_(config.getParameter<double>("mass_lsb")),
        seed_pT_lsb_(config.getParameter<double>("seed_pT_lsb")),
        seed_z0_lsb_(config.getParameter<double>("seed_z0_lsb")),
        sca_sum_lsb_(config.getParameter<double>("sca_sum_lsb")),
        sum_pT_pv_lsb_(config.getParameter<double>("sum_pT_pv_lsb")),
        pos_chg_(config.getParameter<int>("pos_chg")),
        neg_chg_(config.getParameter<int>("neg_chg")) {}

  void L1GTScales::fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<double>("pT_lsb");
    desc.add<double>("phi_lsb");
    desc.add<double>("eta_lsb");
    desc.add<double>("z0_lsb");
    //desc.add<double>("d0_lsb");
    desc.add<double>("isolation_lsb");
    desc.add<double>("beta_lsb");
    desc.add<double>("mass_lsb");
    desc.add<double>("seed_pT_lsb");
    desc.add<double>("seed_z0_lsb");
    desc.add<double>("sca_sum_lsb");
    desc.add<double>("sum_pT_pv_lsb");
    desc.add<int>("pos_chg");
    desc.add<int>("neg_chg");
  }

  PYBIND11_MODULE(libL1TriggerPhase2L1GT, m) {
    py::class_<L1GTScales>(m, "L1GTScales")
        .def(py::init<double,
                      double,
                      double,
                      double,
                      /*double, */
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      int,
                      int>())
        .def("to_hw_pT", &L1GTScales::to_hw_pT)
        .def("to_hw_phi", &L1GTScales::to_hw_phi)
        .def("to_hw_eta", &L1GTScales::to_hw_eta)
        .def("to_hw_z0", &L1GTScales::to_hw_z0)
        .def("to_hw_isolation", &L1GTScales::to_hw_isolation)
        .def("isolation_shift", &L1GTScales::isolation_shift)
        .def("to_hw_beta", &L1GTScales::to_hw_beta)
        .def("to_hw_mass", &L1GTScales::to_hw_mass)
        .def("to_hw_seed_pT", &L1GTScales::to_hw_seed_pT)
        .def("to_hw_seed_z0", &L1GTScales::to_hw_seed_z0)
        .def("to_hw_sca_sum", &L1GTScales::to_hw_sca_sum)
        .def("to_hw_sum_pT_pv", &L1GTScales::to_hw_sum_pT_pv)
        .def("to_hw_dRSquared", &L1GTScales::to_hw_dRSquared)
        .def("to_hw_InvMassSqrDiv2", &L1GTScales::to_hw_InvMassSqrDiv2)
        .def("to_hw_TransMassSqrDiv2", &L1GTScales::to_hw_TransMassSqrDiv2)
        .def("to_hw_PtSquared", &L1GTScales::to_hw_PtSquared)
        .def("neg_chg", &L1GTScales::neg_chg)
        .def("pos_chg", &L1GTScales::pos_chg);
  }
}  // namespace l1t
