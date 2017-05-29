import FWCore.ParameterSet.Config as cms
diphotonEfficiency = cms.EDAnalyzer("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Higgs/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "eff_diphoton_pt          'Photon turnON;             Photon pt [GeV]; efficiency'     photon_pt_numerator          photon_pt_denominator",
        "eff_diphoton_variable 'Photon turnON;             Photon pt [GeV]; efficiency'     photon_pt_variable_numerator photon_pt_variable_denominator",
        "eff_diphoton_eta      'Photon turnON;             Photon eta; efficiency'          photon_eta_numerator         photon_eta_denominator",
        "eff_diphoton_subpt    'Photon turnON;             Photon subpt [GeV]; efficiency'     subphoton_pt_numerator          subphoton_pt_denominator",
        "eff_diphoton_subeta   'Photon turnON;             Photon subeta; efficiency'          subphoton_eta_numerator         subphoton_eta_denominator",
        "eff_diphoton_mass     'Photon turnON;             Diphoton mass; efficiency'          diphoton_mass_numerator         diphoton_mass_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "eff_photon_vs_LS 'Photon pt efficiency vs LS; LS' photonVsLS_numerator photonVsLS_denominator"
    ),
)

higgsClient = cms.Sequence(
    diphotonEfficiency
)
