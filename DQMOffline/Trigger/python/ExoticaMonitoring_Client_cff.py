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

photonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Photon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_photon         'Photon turnON;            Photon pt [GeV]; efficiency'     photon_pt_numerator          photon_pt_denominator",
        "effic_photon_variable 'Photon turnON;            Photon pt [GeV]; efficiency'     photon_pt_variable_numerator photon_pt_variable_denominator",
        "effic_photonPhi       'efficiency vs phi; Photon phi [rad]; efficiency' photon_phi_numerator       photon_phi_denominator",
        "effic_photonEta       'efficiency vs eta; Photon eta; efficiency' photon_eta_numerator       photon_eta_denominator",
        "effic_photonr9       'efficiency vs r9; Photon r9; efficiency' photon_r9_numerator       photon_r9_denominator",
        "effic_photonhoE       'efficiency vs hoE; Photon hoE; efficiency' photon_hoE_numerator       photon_hoE_denominator",
        "effic_photonEtaPhi       'Photon phi; Photon eta; efficiency' photon_etaphi_numerator       photon_etaphi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_photon_vs_LS 'Photon pt efficiency vs LS; LS; PF MET efficiency' photonVsLS_numerator photonVsLS_denominator"
    ),
  
)


exoticaClient = cms.Sequence(
    metEfficiency
    + photonEfficiency
)
