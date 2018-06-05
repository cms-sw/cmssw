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

allEfficiencyPlots = []
add_plot = allEfficiencyPlots.append
for variable, thresholds in variables.iteritems():
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

from DQMOffline.L1Trigger.L1TDiffHarvesting_cfi import l1tDiffHarvesting

resolution_plots = [
    "resolutionMET", "resolutionETMHF", "resolutionPFMetNoMu", "resolutionMHT", "resolutionETT",
    "resolutionHTT", "resolutionMETPhi", "resolutionETMHFPhi", "resolutionPFMetNoMuPhi",
    "resolutionMHTPhi",
]
plots2D = [
    'L1METvsCaloMET', 'L1ETMHFvsCaloETMHF', 'L1METvsPFMetNoMu', 'L1MHTvsRecoMHT', 'L1ETTvsCaloETT',
    'L1HTTvsRecoHTT', 'L1METPhivsCaloMETPhi', 'L1METPhivsPFMetNoMuPhi', 'L1ETMHFPhivsCaloETMHFPhi',
    'L1MHTPhivsRecoMHTPhi',
]

allPlots = []
allPlots.extend(allEfficiencyPlots)
allPlots.extend(resolution_plots)
allPlots.extend(plots2D)

l1tEtSumEmuDiff = l1tDiffHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(  # EMU comparison
            dir1=cms.untracked.string(
                "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco"),
            dir2=cms.untracked.string(
                "L1TEMU/L1TObjects/L1TEtSum/L1TriggerVsReco"),
            outputDir=cms.untracked.string(
                "L1TEMU/L1TObjects/L1TEtSum/L1TriggerVsReco/Comparison"),
            plots=cms.untracked.vstring(allPlots)
        ),
    )
)

# modifications for the pp reference run
variables_HI = variables

allEfficiencyPlots_HI = []
add_plot = allEfficiencyPlots_HI.append
for variable, thresholds in variables_HI.iteritems():
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

allPlots_HI = []
allPlots_HI.extend(allEfficiencyPlots_HI)
allPlots_HI.extend(resolution_plots)
allPlots_HI.extend(plots2D)

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(
    l1tEtSumEmuDiff,
    plotCfgs={0: dict(plots=allPlots_HI)}
)
