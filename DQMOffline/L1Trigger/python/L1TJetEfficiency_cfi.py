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

allEfficiencyPlots = []
add_plot = allEfficiencyPlots.append
for variable, thresholds in variables.iteritems():
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

from DQMOffline.L1Trigger.L1TEfficiencyHarvesting_cfi import l1tEfficiencyHarvesting
l1tJetEfficiency = l1tEfficiencyHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(
            numeratorDir=cms.untracked.string("L1T/L1TObjects/L1TJet/L1TriggerVsReco/efficiency_raw"),
            outputDir=cms.untracked.string("L1T/L1TObjects/L1TJet/L1TriggerVsReco"),
            numeratorSuffix=cms.untracked.string("_Num"),
            denominatorSuffix=cms.untracked.string("_Den"),
            plots=cms.untracked.vstring(allEfficiencyPlots)
        ),
    )
)

l1tJetEmuEfficiency = l1tEfficiencyHarvesting.clone(
    plotCfgs=cms.untracked.VPSet(
        cms.untracked.PSet(
            numeratorDir=cms.untracked.string("L1TEMU/L1TObjects/L1TJet/L1TriggerVsReco/efficiency_raw"),
            outputDir=cms.untracked.string("L1TEMU/L1TObjects/L1TJet/L1TriggerVsReco"),
            numeratorSuffix=cms.untracked.string("_Num"),
            denominatorSuffix=cms.untracked.string("_Den"),
            plots=cms.untracked.vstring(allEfficiencyPlots)
        ),
    )
)

# modifications for the pp reference run
variables_HI = variables
variables_HI['jet'] = L1TStep1.jetEfficiencyThresholds_HI

allEfficiencyPlots_HI = []
add_plot = allEfficiencyPlots_HI.append
for variable, thresholds in variables_HI.iteritems():
    for plot in plots[variable]:
        for threshold in thresholds:
            plotName = '{0}_threshold_{1}'.format(plot, threshold)
            add_plot(plotName)

from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
ppRef_2017.toModify(l1tJetEfficiency,
    plotCfgs = {
        0:dict(plots = allEfficiencyPlots_HI),
    }
)
ppRef_2017.toModify(l1tJetEmuEfficiency,
    plotCfgs = {
        0:dict(plots = allEfficiencyPlots_HI),
    }
)
