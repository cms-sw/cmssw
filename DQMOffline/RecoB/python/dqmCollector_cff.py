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

bTagCollectorMC = pfbTagAnalysis.clone(
    finalizePlots = True,
    finalizeOnly = True,
    mcPlots = 1,
)
# module execution
bTagCollectorSequenceMC = cms.Sequence(bTagCollectorMC)
#### end ####
