import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TEtSumJetOffline_cfi as L1TStep1
import six

variables = {
    'jet': L1TStep1.jetEfficiencyThresholds,
}

plots = {
    'jet': [
        "efficiencyJetEt_HB", "efficiencyJetEt_HE", "efficiencyJetEt_HF",
        "efficiencyJetEt_HB_HE"],
}

allEfficiencyPlots = []
add_plot = allEfficiencyPlots.append
for variable, thresholds in six.iteritems(variables):
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

from DQMOffline.L1Trigger.L1TDiffHarvesting_cfi import l1tDiffHarvesting

resolution_plots = [
    "resolutionJetET_HB", "resolutionJetET_HE", "resolutionJetET_HF",
    "resolutionJetET_HB_HE", "resolutionJetPhi_HB", "resolutionJetPhi_HE",
    "resolutionJetPhi_HF", "resolutionJetPhi_HB_HE", "resolutionJetEta",
]
plots2D = [
    'L1JetETvsCaloJetET_HB', 'L1JetETvsCaloJetET_HE', 'L1JetETvsCaloJetET_HF',
    'L1JetETvsCaloJetET_HB_HE', 'L1JetPhivsCaloJetPhi_HB', 'L1JetPhivsCaloJetPhi_HE',
    'L1JetPhivsCaloJetPhi_HF', 'L1JetPhivsCaloJetPhi_HB_HE', 'L1JetEtavsCaloJetEta_HB',
]

allPlots = []
allPlots.extend(allEfficiencyPlots)
allPlots.extend(resolution_plots)
allPlots.extend(plots2D)

l1tJetEmuDiff = l1tDiffHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(  # EMU comparison
            dir1=cms.untracked.string("L1T/L1TObjects/L1TJet/L1TriggerVsReco"),
            dir2=cms.untracked.string("L1TEMU/L1TObjects/L1TJet/L1TriggerVsReco"),
            outputDir=cms.untracked.string(
                "L1TEMU/L1TObjects/L1TJet/L1TriggerVsReco/Comparison"),
            plots=cms.untracked.vstring(allPlots)
        ),
    )
)

# modifications for the pp reference run
variables_HI = variables
variables_HI['jet'] = L1TStep1.jetEfficiencyThresholds_HI

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
ppRef_2017.toModify(l1tJetEmuDiff,
    plotCfgs = {0:dict(plots = allPlots_HI)}
)

