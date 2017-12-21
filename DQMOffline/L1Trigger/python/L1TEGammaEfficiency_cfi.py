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

# modifications for the pp reference run
variables_HI = {
    'electron': L1TEGammaOffline_cfi.electronEfficiencyThresholds_HI,
    'photon': L1TEGammaOffline_cfi.photonEfficiencyThresholds_HI,
}

deepInspectionThresholds_HI = {
    'electron': L1TEGammaOffline_cfi.deepInspectionElectronThresholds_HI,
    'photon': [],
}

allEfficiencyPlots_HI = []
add_plot = allEfficiencyPlots_HI.append
for variable, thresholds in variables_HI.iteritems():
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

for variable, thresholds in deepInspectionThresholds_HI.iteritems():
    for plot in deepInspectionPlots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tEGammaEfficiency,
    plotCfgs = {
        0:dict(plots = allEfficiencyPlots_HI),
        1:dict(plots = allEfficiencyPlots_HI)
    }
)
