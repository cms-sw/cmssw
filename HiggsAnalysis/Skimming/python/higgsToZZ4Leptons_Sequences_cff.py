import FWCore.ParameterSet.Config as cms

# HLT analysis
from  HiggsAnalysis.Skimming.higgsToZZ4LeptonsHLTAnalysis_cfi import *

# leptons producer
from HiggsAnalysis.Skimming.higgsToZZ4LeptonsBuildLeptons_cfi import *
#import HiggsAnalysis.Skimming.higgsToZZ4LeptonsBuildLeptons

# DiLeptons
from HiggsAnalysis.Skimming.higgsToZZ4LeptonsSkimDiLeptonProducer_cfi import *
#import HiggsAnalysis.Skimming.higgsToZZ4LeptonsSkimDiLeptonProducer

# TriLeptons
from HiggsAnalysis.Skimming.higgsToZZ4LeptonsSkimTriLeptonProducer_cfi import *
#import HiggsAnalysis.Skimming.higgsToZZ4LeptonsSkimTriLeptonProducer_cfi

# HZZ4l filter
from HiggsAnalysis.Skimming.higgsToZZ4LeptonsSkimFilter_cfi import *
import HiggsAnalysis.Skimming.higgsToZZ4LeptonsSkimFilter_cfi

# HLT trig report
from HLTrigger.HLTanalyzers.hlTrigReport_cfi import *
import HLTrigger.HLTanalyzers.hlTrigReport_cfi
hltReport = HLTrigger.HLTanalyzers.hlTrigReport_cfi.hlTrigReport.clone()
hltReport.HLTriggerResults = cms.InputTag("TriggerResults","","HLT8E29")

higgsToZZ4LeptonsSequence = cms.Sequence(
        higgsToZZ4LeptonsHLTAnalysis  +
        higgsToZZ4LeptonsBuildLeptons          +
	higgsToZZ4LeptonsSkimDiLeptonProducer  +
        higgsToZZ4LeptonsSkimTriLeptonProducer +
        higgsToZZ4LeptonsSkimFilter   
	)

