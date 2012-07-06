import FWCore.ParameterSet.Config as cms

from RecoBTag.Configuration.RecoBTag_cff import *

from DQMOffline.RecoB.dqmAnalyzer_cff import *
bTagCollector = pfbTagAnalysis.clone()

# module execution
bTagCollectorSequence = cms.Sequence(bTagCollector)
bTagCollector.finalizePlots = True
bTagCollector.finalizeOnly = True


