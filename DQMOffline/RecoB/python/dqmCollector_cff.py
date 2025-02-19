import FWCore.ParameterSet.Config as cms

from RecoBTag.Configuration.RecoBTag_cff import *
import DQMOffline.RecoB.bTagAnalysisData_cfi
bTagCollector = DQMOffline.RecoB.bTagAnalysisData_cfi.bTagAnalysis.clone()
# module execution
bTagCollectorSequence = cms.Sequence(bTagCollector)
bTagCollector.finalizePlots = True
bTagCollector.finalizeOnly = True


