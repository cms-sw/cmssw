import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TEGammaOffline_cfi

variables = {
    'electron': L1TEGammaOffline_cfi.electronEfficiencyThresholds,
}

plots = {
    'electron': [
        "efficiencyElectronET_EB", "efficiencyElectronET_EE",
        "efficiencyElectronET_EB_EE"
    ],
}

allEfficiencyPlots = []
add_plot = allEfficiencyPlots.append
for variable, thresholds in variables.iteritems():
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

resolution_plots = [
    "resolutionElectronET_EB", "resolutionElectronET_EE",
    "resolutionElectronET_EB_EE", "resolutionElectronPhi_EB", "resolutionElectronPhi_EE",
    "resolutionElectronPhi_EB_EE", "resolutionElectronEta",
]
plots2D = [
    'L1EGammaETvsElectronET_EB', 'L1EGammaETvsElectronET_EE', 'L1EGammaETvsElectronET_EB_EE',
    'L1EGammaPhivsElectronPhi_EB', 'L1EGammaPhivsElectronPhi_EE', 'L1EGammaPhivsElectronPhi_EB_EE',
    'L1EGammaEtavsElectronEta',
]

allPlots = []
allPlots.extend(allEfficiencyPlots)
allPlots.extend(resolution_plots)
allPlots.extend(plots2D)

from DQMOffline.L1Trigger.L1TDiffHarvesting_cfi import l1tDiffHarvesting
l1tEGammaEmuDiff = l1tDiffHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(  # EMU comparison
            dir1=cms.untracked.string("L1T/L1TEGamma"),
            dir2=cms.untracked.string("L1TEMU/L1TEGamma"),
            outputDir=cms.untracked.string(
                "L1TEMU/L1TEGamma/Comparison"),
            plots=cms.untracked.vstring(allPlots)
        ),
    )
)
