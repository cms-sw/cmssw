import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TEtSumJetOffline_cfi as L1TStep1

variables = {
    'jet': L1TStep1.jetEfficiencyThresholds,
}

plots = {
    'jet': [
        "efficiencyJetEt_HB", "efficiencyJetEt_HE", "efficiencyJetEt_HF",
        "efficiencyJetEt_HB_HE"],
}

from DQMOffline.L1Trigger.L1TCommon import generateEfficiencyStrings
efficiencyStrings = list(generateEfficiencyStrings(variables, plots))

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
l1tJetEfficiency = DQMEDHarvester(
    "DQMGenericClient",
    commands=cms.vstring(),
    resolution=cms.vstring(),
    subDirs=cms.untracked.vstring('L1T/L1TObjects/L1TJet/L1TriggerVsReco'),
    efficiency=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(efficiencyStrings),
)

l1tJetEmuEfficiency = l1tJetEfficiency.clone(
    subDirs=cms.untracked.vstring(
        'L1TEMU/L1TObjects/L1TJet/L1TriggerVsReco'),
)

# modifications for the pp reference run
variables_HI = variables
variables_HI['jet'] = L1TStep1.jetEfficiencyThresholds_HI

efficiencyStrings_HI = list(generateEfficiencyStrings(variables_HI, plots))

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tJetEfficiency, efficiencyProfile=efficiencyStrings_HI)
ppRef_2017.toModify(l1tJetEmuEfficiency, efficiencyProfile=efficiencyStrings_HI)
