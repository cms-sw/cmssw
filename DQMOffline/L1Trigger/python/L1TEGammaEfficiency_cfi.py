import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TEGammaOffline_cfi

variables = {
    'electron': L1TEGammaOffline_cfi.electronEfficiencyThresholds,
    'photon': L1TEGammaOffline_cfi.photonEfficiencyThresholds,
}

deepInspectionThresholds = {
    'electron': L1TEGammaOffline_cfi.deepInspectionElectronThresholds,
    'photon': [],
}

plots = {
    'electron': [
        "efficiencyElectronET_EB", "efficiencyElectronET_EE",
        "efficiencyElectronET_EB_EE", "efficiencyElectronPhi_vs_Eta",
    ],
    'photon': [
        "efficiencyPhotonET_EB", "efficiencyPhotonET_EE",
        "efficiencyPhotonET_EB_EE"
    ]
}

deepInspectionPlots = {
    'electron': [
        'efficiencyElectronEta', 'efficiencyElectronPhi',
        'efficiencyElectronNVertex'
    ],
    'photon': [],
}

variables_HI = {
    'electron': L1TEGammaOffline_cfi.electronEfficiencyThresholds_HI,
    'photon': L1TEGammaOffline_cfi.photonEfficiencyThresholds_HI,
}

deepInspectionThresholds_HI = {
    'electron': L1TEGammaOffline_cfi.deepInspectionElectronThresholds_HI,
    'photon': [],
}


# remove photon variables (code to produce them is currently commented out)
variables['photon'] = []
variables_HI['photon'] = []

from DQMOffline.L1Trigger.L1TCommon import generateEfficiencyStrings as ges
efficiencyStrings = list(ges(variables, plots))
efficiencyStrings += list(ges(deepInspectionThresholds, deepInspectionPlots))

efficiencyStrings_HI = list(ges(variables_HI, plots))
efficiencyStrings_HI += list(ges(deepInspectionThresholds_HI,
                                 deepInspectionPlots))

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
l1tEGammaEfficiency = DQMEDHarvester(
    "DQMGenericClient",
    commands=cms.vstring(),
    resolution=cms.vstring(),
    subDirs=cms.untracked.vstring('L1T/L1TObjects/L1TEGamma/L1TriggerVsReco'),
    efficiency=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(efficiencyStrings),
)

l1tEGammaEmuEfficiency = l1tEGammaEfficiency.clone(
    subDirs=cms.untracked.vstring(
        'L1TEMU/L1TObjects/L1TEGamma/L1TriggerVsReco'),
)

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tEGammaEfficiency, efficiencyProfile=efficiencyStrings_HI)
ppRef_2017.toModify(l1tEGammaEmuEfficiency, efficiencyProfile=efficiencyStrings_HI)
