import FWCore.ParameterSet.Config as cms

diphotonEfficiency = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Higgs/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "eff_diphoton         'Photon turnON;             Photon pt [GeV]; efficiency'     photon_pt_numerator          photon_pt_denominator",
        "eff_diphoton_variable 'Photon turnON;            Photon pt [GeV]; efficiency'     photon_pt_variable_numerator photon_pt_variable_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "eff_photon_vs_LS 'Photon pt efficiency vs LS; LS' photonVsLS_numerator photonVsLS_denominator"
    ),
)

higgsClient = cms.Sequence(
    diphotonEfficiency
    )
