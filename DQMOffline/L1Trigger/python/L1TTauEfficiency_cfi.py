import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TTauOffline_cfi

variables = {
    'tau': L1TTauOffline_cfi.tauEfficiencyThresholds,
    'NonIsotau': L1TTauOffline_cfi.tauEfficiencyThresholds,
}

plots = {
    'tau': [
        "efficiencyIsoTauET_EB", "efficiencyIsoTauET_EE",
        "efficiencyIsoTauET_EB_EE"
    ],
    'NonIsotau' : [
        "efficiencyNonIsoTauET_EB", "efficiencyNonIsoTauET_EE",
        "efficiencyNonIsoTauET_EB_EE"
    ]
}

from DQMOffline.L1Trigger.L1TCommon import generateEfficiencyStrings

efficiencyStrings = list(generateEfficiencyStrings(variables, plots))


from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
l1tTauEfficiency = DQMEDHarvester(
    "DQMGenericClient",
    commands=cms.vstring(),
    resolution=cms.vstring(),
    subDirs=cms.untracked.vstring('L1T/L1TObjects/L1TTau/L1TriggerVsReco'),
    efficiency=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(efficiencyStrings),
)

l1tTauEmuEfficiency = l1tTauEfficiency.clone(
    subDirs=cms.untracked.vstring(
        'L1TEMU/L1TObjects/L1TTau/L1TriggerVsReco'),
)
