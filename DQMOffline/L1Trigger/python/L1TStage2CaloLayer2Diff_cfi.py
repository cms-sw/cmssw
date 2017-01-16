import FWCore.ParameterSet.Config as cms
from DQMOffline.L1Trigger import L1TStage2CaloLayer2Offline_cfi as L1TStep1

variables = {
    'jet': L1TStep1.jetEfficiencyThresholds,
    'met': L1TStep1.metEfficiencyThresholds,
    'mht': L1TStep1.mhtEfficiencyThresholds,
    'ett': L1TStep1.ettEfficiencyThresholds,
    'htt': L1TStep1.httEfficiencyThresholds,
}

plots = {
    'jet': [
        "efficiencyJetEt_HB", "efficiencyJetEt_HE", "efficiencyJetEt_HF",
        "efficiencyJetEt_HB_HE"],
    'met': ['efficiencyMET'],
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
    "resolutionJetET_HB", "resolutionJetET_HE", "resolutionJetET_HF",
    "resolutionJetET_HB_HE", "resolutionJetPhi_HB", "resolutionJetPhi_HE",
    "resolutionJetPhi_HF", "resolutionJetPhi_HB_HE", "resolutionJetEta",
    "resolutionMET", "resolutionMHT", "resolutionETT", "resolutionHTT",
    "resolutionMETPhi", "resolutionMHTPhi",
]
plots2D = [
    "L1METvsCaloMET", 'L1MHTvsRecoMHT', 'L1ETTvsCaloETT', 'L1HTTvsRecoHTT',
    'L1METPhivsCaloMETPhi', 'L1MHTPhivsRecoMHTPhi',
    # jets
    'L1JetETvsCaloJetET_HB', 'L1JetETvsCaloJetET_HE', 'L1JetETvsCaloJetET_HF',
    'L1JetETvsCaloJetET_HB_HE', 'L1JetETvsCaloJetET_HB', 'L1JetETvsCaloJetET_HE',
    'L1JetETvsCaloJetET_HF', 'L1JetETvsCaloJetET_HB_HE', 'L1JetEtavsCaloJetEta_HB',
]

allPlots = []
allPlots.extend(allEfficiencyPlots)
allPlots.extend(resolution_plots)
allPlots.extend(plots2D)

l1tStage2CaloLayer2EmuDiff = l1tDiffHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(  # EMU comparison
            dir1=cms.untracked.string("L1T/L1TStage2CaloLayer2"),
            dir2=cms.untracked.string("L1TEMU/L1TStage2CaloLayer2"),
            outputDir=cms.untracked.string(
                "L1TEMU/L1TStage2CaloLayer2/Comparison"),
            plots=cms.untracked.vstring(allPlots)
        ),
    )
)
