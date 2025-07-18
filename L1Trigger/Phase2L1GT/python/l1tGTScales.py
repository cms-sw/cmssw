import FWCore.ParameterSet.Config as cms
import math

scale_parameter = cms.PSet(
    pT_lsb=cms.double(0.03125),            # GeV
    phi_lsb=cms.double(math.pi / 2**12),   # radiants
    eta_lsb=cms.double(math.pi / 2**12),   # radiants
    z0_lsb=cms.double(1/(5*2**9)),         # cm
    # d0_lsb = cms.double(...), TODO input scales far apart
    isolationPT_lsb=cms.double(0.25),      # GeV
    beta_lsb=cms.double(1. / 2**4),        # [0, 1]
    mass_lsb=cms.double(0.25),             # GeV^2
    seed_pT_lsb=cms.double(0.25),          # GeV
    seed_z0_lsb=cms.double(30. / 2**9),    # ? cm
    scalarSumPT_lsb=cms.double(0.03125),   # GeV
    sum_pT_pv_lsb=cms.double(0.25),        # GeV
    pos_chg=cms.int32(0),
    neg_chg=cms.int32(1)
)
