import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

bphEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/BPH/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPhi       'mu efficiency vs phi; mu phi [rad]; efficiency' muPhi_numerator       muPhi_denominator",
        "effic_muEta       'mu efficiency vs eta; mu eta [rad]; efficiency' muEta_numerator       muEta_denominator",
        "effic_muPt       'mu efficiency vs pt; mu pt [GeV]; efficiency' muPt_numerator       muPt_denominator",
        "effic_mu_d0       'mu efficiency vs d0; mu d0 [cm]; efficiency' mu_d0_numerator       mu_d0_denominator",
        "effic_mu_z0       'mu efficiency vs z0; mu z0 [cm]; efficiency' mu_z0_numerator       mu_z0_denominator",
    ),
#    efficiencyProfile = cms.untracked.vstring(
#        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator"
#    ),
  
)

bphClient = cms.Sequence(
    bphEfficiency
)
