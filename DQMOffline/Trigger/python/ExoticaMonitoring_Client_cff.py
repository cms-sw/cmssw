import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

metEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/MET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met          'MET turnON;            PF MET [GeV]; efficiency'     met_numerator          met_denominator",
        "effic_met_variable 'MET turnON;            PF MET [GeV]; efficiency'     met_variable_numerator met_variable_denominator",
        "effic_metPhi       'MET efficiency vs phi; PF MET phi [rad]; efficiency' metPhi_numerator       metPhi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator"
    ),
  
)

exoticaClient = cms.Sequence(
    metEfficiency
)
