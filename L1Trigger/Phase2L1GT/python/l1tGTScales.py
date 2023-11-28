from libL1TriggerPhase2L1GT import L1GTScales as CppScales
import FWCore.ParameterSet.Config as cms
import math

scale_parameter = cms.PSet(
    pT_lsb=cms.double(0.03125),            # GeV
    phi_lsb=cms.double(math.pi / 2**12),   # radiants
    eta_lsb=cms.double(math.pi / 2**12),   # radiants
    z0_lsb=cms.double(1/(5*2**9)),         # cm
    # d0_lsb = cms.double(...), TODO input scales far apart
    isolation_lsb=cms.double(0.25),        # GeV
    beta_lsb=cms.double(1. / 2**4),        # [0, 1]
    mass_lsb=cms.double(0.25),             # GeV^2
    seed_pT_lsb=cms.double(0.25),          # GeV
    seed_z0_lsb=cms.double(30. / 2**9),    # ? cm
    sca_sum_lsb=cms.double(0.03125),       # GeV
    sum_pT_pv_lsb=cms.double(0.25),        # GeV
    pos_chg=cms.int32(1),
    neg_chg=cms.int32(0)
)

l1tGTScales = CppScales(*[param.value() for param in scale_parameter.parameters_().values()])
