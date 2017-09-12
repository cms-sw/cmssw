import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TTauOffline_cfi

variables = {
    'tau': L1TTauOffline_cfi.tauEfficiencyThresholds,
}

plots = {
    'tau': [
        "efficiencyIsoTauET_EB", "efficiencyIsoTauET_EE",
        "efficiencyIsoTauET_EB_EE"
    ],
    'NonIsotau': [
        "efficiencyNonIsoTauET_EB", "efficiencyNonIsoTauET_EE",
        "efficiencyNonIsoTauET_EB_EE"
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
    "resolutionTauET_EB", "resolutionTauET_EE",
    "resolutionTauET_EB_EE", "resolutionTauPhi_EB", "resolutionTauPhi_EE",
    "resolutionTauPhi_EB_EE", "resolutionTauEta",

]
plots2D = [
    'L1TauETvsTauET_EB', 'L1TauETvsTauET_EE', 'L1TauETvsTauET_EB_EE',
    'L1TauPhivsTauPhi_EB', 'L1TauPhivsTauPhi_EE', 'L1TauPhivsTauPhi_EB_EE',
    'L1TauEtavsTauEta',
]

allPlots = []
allPlots.extend(allEfficiencyPlots)
allPlots.extend(resolution_plots)
allPlots.extend(plots2D)

from DQMOffline.L1Trigger.L1TDiffHarvesting_cfi import l1tDiffHarvesting
l1tTauEmuDiff = l1tDiffHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(  # EMU comparison
            dir1=cms.untracked.string("L1T/L1TTau"),
            dir2=cms.untracked.string("L1TEMU/L1TTau"),
            outputDir=cms.untracked.string(
                "L1TEMU/L1TTau/Comparison"),
            plots=cms.untracked.vstring(allPlots)
        ),
    )
)
