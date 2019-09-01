import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TEGammaOffline_cfi
import six

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
for variable, thresholds in six.iteritems(variables):
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

resolution_plots = [
    "resolutionElectronET_EB", "resolutionElectronET_EE",
    "resolutionElectronET_EB_EE", "resolutionElectronPhi_EB", "resolutionElectronPhi_EE",
    "resolutionElectronPhi_EB_EE", "resolutionElectronEta",
    #
    "resolutionPhotonET_EB", "resolutionPhotonET_EE",
    "resolutionPhotonET_EB_EE", "resolutionPhotonPhi_EB", "resolutionPhotonPhi_EE",
    "resolutionPhotonPhi_EB_EE", "resolutionPhotonEta",
]
plots2D = [
    'L1EGammaETvsElectronET_EB', 'L1EGammaETvsElectronET_EE', 'L1EGammaETvsElectronET_EB_EE',
    'L1EGammaPhivsElectronPhi_EB', 'L1EGammaPhivsElectronPhi_EE', 'L1EGammaPhivsElectronPhi_EB_EE',
    'L1EGammaEtavsElectronEta',
    #
    'L1EGammaETvsPhotonET_EB', 'L1EGammaETvsPhotonET_EE', 'L1EGammaETvsPhotonET_EB_EE',
    'L1EGammaPhivsPhotonPhi_EB', 'L1EGammaPhivsPhotonPhi_EE', 'L1EGammaPhivsPhotonPhi_EB_EE',
    'L1EGammaEtavsPhotonEta',
]

# remove photon variables (code to produce them is currently commented out)
resolution_plots = [plot for plot in resolution_plots if 'Photon' not in plot]
plots2D = [plot for plot in plots2D if 'Photon' not in plot]

allPlots = []
allPlots.extend(allEfficiencyPlots)
allPlots.extend(resolution_plots)
allPlots.extend(plots2D)


from DQMOffline.L1Trigger.L1TDiffHarvesting_cfi import l1tDiffHarvesting
l1tEGammaEmuDiff = l1tDiffHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(  # EMU comparison
            dir1=cms.untracked.string("L1T/L1TObjects/L1TEGamma/L1TriggerVsReco"),
            dir2=cms.untracked.string("L1TEMU/L1TObjects/L1TEGamma/L1TriggerVsReco"),
            outputDir=cms.untracked.string(
                "L1TEMU/L1TObjects/L1TEGamma/L1TriggerVsReco/Comparison"),
            plots=cms.untracked.vstring(allPlots)
        ),
    )
)

# modifications for the pp reference run
variables_HI = {
    'electron': L1TEGammaOffline_cfi.electronEfficiencyThresholds_HI,
}

allEfficiencyPlots_HI = []
add_plot = allEfficiencyPlots_HI.append
for variable, thresholds in six.iteritems(variables_HI):
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

allPlots_HI = []
allPlots_HI.extend(allEfficiencyPlots_HI)
allPlots_HI.extend(resolution_plots)
allPlots_HI.extend(plots2D)

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tEGammaEmuDiff,
    plotCfgs = {0:dict(plots = allPlots_HI)}
)
