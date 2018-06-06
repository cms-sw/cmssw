import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TEtSumJetOffline_cfi as L1TStep1

variables = {
    'met': L1TStep1.metEfficiencyThresholds,
    'mht': L1TStep1.mhtEfficiencyThresholds,
    'ett': L1TStep1.ettEfficiencyThresholds,
    'htt': L1TStep1.httEfficiencyThresholds,
}

plots = {
    'met': ['efficiencyMET', 'efficiencyETMHF', 'efficiencyPFMetNoMu'],
    'mht': ['efficiencyMHT'],
    'ett': ['efficiencyETT'],
    'htt': ['efficiencyHTT'],
}

from DQMOffline.L1Trigger.L1TCommon import generateEfficiencyStrings

efficiencyStrings = list(generateEfficiencyStrings(variables, plots))

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
l1tEtSumEfficiency = DQMEDHarvester(
    "DQMGenericClient",
    commands=cms.vstring(),
    resolution=cms.vstring(),
    subDirs=cms.untracked.vstring('L1T/L1TObjects/L1TEtSum/L1TriggerVsReco'),
    efficiency=cms.vstring(),
    efficiencyProfile=cms.untracked.vstring(efficiencyStrings),
)

l1tEtSumEmuEfficiency = l1tEtSumEfficiency.clone(
    subDirs=cms.untracked.vstring(
        'L1TEMU/L1TObjects/L1TEtSum/L1TriggerVsReco'),
)

# modifications for the pp reference run
variables_HI = variables
efficiencyStrings_HI = list(generateEfficiencyStrings(variables_HI, plots))

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tEtSumEfficiency, efficiencyProfile=efficiencyStrings_HI)
ppRef_2017.toModify(l1tEtSumEmuEfficiency, efficiencyProfile=efficiencyStrings_HI)
