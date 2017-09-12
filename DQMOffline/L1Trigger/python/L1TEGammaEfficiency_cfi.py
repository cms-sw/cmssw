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

allEfficiencyPlots = []
add_plot = allEfficiencyPlots.append
for variable, thresholds in variables.iteritems():
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

for variable, thresholds in deepInspectionThresholds.iteritems():
    for plot in deepInspectionPlots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

from DQMOffline.L1Trigger.L1TEfficiencyHarvesting_cfi import l1tEfficiencyHarvesting
l1tEGammaEfficiency = l1tEfficiencyHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(
            numeratorDir=cms.untracked.string("L1T/L1TEGamma/efficiency_raw"),
            outputDir=cms.untracked.string("L1T/L1TEGamma"),
            numeratorSuffix=cms.untracked.string("_Num"),
            denominatorSuffix=cms.untracked.string("_Den"),
            plots=cms.untracked.vstring(allEfficiencyPlots)
        ),
        cms.untracked.PSet(
            numeratorDir=cms.untracked.string(
                "L1TEMU/L1TEGamma/efficiency_raw"),
            outputDir=cms.untracked.string("L1TEMU/L1TEGamma"),
            numeratorSuffix=cms.untracked.string("_Num"),
            denominatorSuffix=cms.untracked.string("_Den"),
            plots=cms.untracked.vstring(allEfficiencyPlots)
        ),
    )
)
