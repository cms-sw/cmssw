import FWCore.ParameterSet.Config as cms

from RecoBTag.Configuration.RecoBTag_cff import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *

calobTagCollector = calobTagAnalysis.clone()
# module execution
bTagCollectorSequence = cms.Sequence(calobTagCollector)
calobTagCollector.finalizePlots = True
calobTagCollector.finalizeOnly = True

bTagCollectorDATA = pfbTagAnalysis.clone()
# module execution
bTagCollectorSequenceDATA = cms.Sequence(bTagCollectorDATA)
bTagCollectorDATA.finalizePlots = True
bTagCollectorDATA.finalizeOnly = True

bTagCollectorMC = pfbTagValidation.clone()
# module execution
bTagCollectorSequenceMC = cms.Sequence(bTagCollectorMC)
bTagCollectorMC.finalizePlots = True
bTagCollectorMC.finalizeOnly = True
