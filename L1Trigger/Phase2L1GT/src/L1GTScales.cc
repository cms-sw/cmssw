#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace l1t {
  L1GTScales::L1GTScales(double pT_lsb,
                         double phi_lsb,
                         double eta_lsb,
                         double z0_lsb,
                         //double d0_lsb,
                         double isolationPT_lsb,
                         double beta_lsb,
                         double mass_lsb,
                         double seed_pT_lsb,
                         double seed_z0_lsb,
                         double scalarSumPT_lsb,
                         double sum_pT_pv_lsb,
                         int pos_chg,
                         int neg_chg)
      : pT_lsb_(pT_lsb),
        phi_lsb_(phi_lsb),
        eta_lsb_(eta_lsb),
        z0_lsb_(z0_lsb),
        //d0_lsb_(d0_lsb),
        isolationPT_lsb_(isolationPT_lsb),
        beta_lsb_(beta_lsb),
        mass_lsb_(mass_lsb),
        seed_pT_lsb_(seed_pT_lsb),
        seed_z0_lsb_(seed_z0_lsb),
        scalarSumPT_lsb_(scalarSumPT_lsb),
        sum_pT_pv_lsb_(sum_pT_pv_lsb),
        pos_chg_(pos_chg),
        neg_chg_(neg_chg) {}

  L1GTScales::L1GTScales(const edm::ParameterSet& config)
      : pT_lsb_(config.getParameter<double>("pT_lsb")),
        phi_lsb_(config.getParameter<double>("phi_lsb")),
        eta_lsb_(config.getParameter<double>("eta_lsb")),
        z0_lsb_(config.getParameter<double>("z0_lsb")),
        //d0_lsb_(config.getParameter<double>("d0_lsb")),
        isolationPT_lsb_(config.getParameter<double>("isolationPT_lsb")),
        beta_lsb_(config.getParameter<double>("beta_lsb")),
        mass_lsb_(config.getParameter<double>("mass_lsb")),
        seed_pT_lsb_(config.getParameter<double>("seed_pT_lsb")),
        seed_z0_lsb_(config.getParameter<double>("seed_z0_lsb")),
        scalarSumPT_lsb_(config.getParameter<double>("scalarSumPT_lsb")),
        sum_pT_pv_lsb_(config.getParameter<double>("sum_pT_pv_lsb")),
        pos_chg_(config.getParameter<int>("pos_chg")),
        neg_chg_(config.getParameter<int>("neg_chg")) {}

  void L1GTScales::fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<double>("pT_lsb");
    desc.add<double>("phi_lsb");
    desc.add<double>("eta_lsb");
    desc.add<double>("z0_lsb");
    //desc.add<double>("d0_lsb");
    desc.add<double>("isolationPT_lsb");
    desc.add<double>("beta_lsb");
    desc.add<double>("mass_lsb");
    desc.add<double>("seed_pT_lsb");
    desc.add<double>("seed_z0_lsb");
    desc.add<double>("scalarSumPT_lsb");
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
        .def("to_hw_pT_ceil", &L1GTScales::to_hw_pT_ceil)
        .def("to_hw_phi_ceil", &L1GTScales::to_hw_phi_ceil)
        .def("to_hw_eta_ceil", &L1GTScales::to_hw_eta_ceil)
        .def("to_hw_z0_ceil", &L1GTScales::to_hw_z0_ceil)
        .def("to_hw_isolationPT_ceil", &L1GTScales::to_hw_isolationPT_ceil)
        .def("to_hw_relative_isolationPT_ceil", &L1GTScales::to_hw_relative_isolationPT_ceil)
        .def("to_hw_beta_ceil", &L1GTScales::to_hw_beta_ceil)
        .def("to_hw_mass_ceil", &L1GTScales::to_hw_mass_ceil)
        .def("to_hw_seed_pT_ceil", &L1GTScales::to_hw_seed_pT_ceil)
        .def("to_hw_seed_z0_ceil", &L1GTScales::to_hw_seed_z0_ceil)
        .def("to_hw_scalarSumPT_ceil", &L1GTScales::to_hw_scalarSumPT_ceil)
        .def("to_hw_sum_pT_pv_ceil", &L1GTScales::to_hw_sum_pT_pv_ceil)
        .def("to_hw_dRSquared_ceil", &L1GTScales::to_hw_dRSquared_ceil)
        .def("to_hw_pT_floor", &L1GTScales::to_hw_pT_floor)
        .def("to_hw_phi_floor", &L1GTScales::to_hw_phi_floor)
        .def("to_hw_eta_floor", &L1GTScales::to_hw_eta_floor)
        .def("to_hw_z0_floor", &L1GTScales::to_hw_z0_floor)
        .def("to_hw_isolationPT_floor", &L1GTScales::to_hw_isolationPT_floor)
        .def("to_hw_relative_isolationPT_floor", &L1GTScales::to_hw_relative_isolationPT_floor)
        .def("to_hw_beta_floor", &L1GTScales::to_hw_beta_floor)
        .def("to_hw_mass_floor", &L1GTScales::to_hw_mass_floor)
        .def("to_hw_seed_pT_floor", &L1GTScales::to_hw_seed_pT_floor)
        .def("to_hw_seed_z0_floor", &L1GTScales::to_hw_seed_z0_floor)
        .def("to_hw_scalarSumPT_floor", &L1GTScales::to_hw_scalarSumPT_floor)
        .def("to_hw_sum_pT_pv_floor", &L1GTScales::to_hw_sum_pT_pv_floor)
        .def("to_hw_dRSquared_floor", &L1GTScales::to_hw_dRSquared_floor)
        .def("to_hw_InvMassSqrDiv2", &L1GTScales::to_hw_InvMassSqrDiv2)
        .def("to_hw_TransMassSqrDiv2", &L1GTScales::to_hw_TransMassSqrDiv2)
        .def("to_hw_PtSquared", &L1GTScales::to_hw_PtSquared)
        .def("to_hw_InvMassSqrOver2DR", &L1GTScales::to_hw_InvMassSqrOver2DR)
        .def("neg_chg", &L1GTScales::neg_chg)
        .def("pos_chg", &L1GTScales::pos_chg);
  }
}  // namespace l1t
