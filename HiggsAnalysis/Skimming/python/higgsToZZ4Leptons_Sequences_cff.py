import FWCore.ParameterSet.Config as cms

# HLT filter with paths
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_HLTPaths_cfi import *

# HLT analysis
from  HiggsAnalysis.HiggsToZZ4Leptons.hTozzTo4leptonsHLTAnalysis_cfi import *

# HZZ4l filter
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_Filter_cfi import *
import HiggsAnalysis.Skimming.higgsToZZ4Leptons_Filter_cfi 
higgsToZZ4LeptonsFilterNew=HiggsAnalysis.Skimming.higgsToZZ4Leptons_Filter_cfi.higgsToZZ4LeptonsFilter.clone()

# HLT trig report
from HLTrigger.HLTanalyzers.hlTrigReport_cfi import *
import HLTrigger.HLTanalyzers.hlTrigReport_cfi
hltReport = HLTrigger.HLTanalyzers.hlTrigReport_cfi.hlTrigReport.clone()
hltReport.HLTriggerResults = cms.InputTag("TriggerResults","","HLTreprocess")

higgsToZZ4LeptonsSequence = cms.Sequence(
        hTozzTo4leptonsHLTAnalysis +
	higgsToZZ4LeptonsHLTFilter +
	higgsToZZ4LeptonsFilterNew +
        hltReport 
	)

