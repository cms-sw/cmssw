import FWCore.ParameterSet.Config as cms

from RecoBTag.Configuration.RecoBTag_cff import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *

#pf DATA collector
bTagCollectorDATA = bTagAnalysis.clone()
# module execution
bTagCollectorSequenceDATA = cms.Sequence(bTagCollectorDATA)
bTagCollectorDATA.ptRanges = cms.vdouble(0.0)
bTagCollectorDATA.finalizePlots = True
bTagCollectorDATA.finalizeOnly = True


#pf MC collector
bTagCollectorMC = bTagAnalysis.clone(
    finalizePlots = True,
    finalizeOnly = True,
    mcPlots = 2, #harvest all, b, c, dusg and ni histos 
    ptRanges = cms.vdouble(0.0),
    etaRanges = cms.vdouble(0.0),
)
# module execution
bTagCollectorSequenceMC = cms.Sequence(bTagCollectorMC)

#special sequence for fullsim, all histos havested by the DATA sequence in the dqm offline sequence
bTagCollectorMCbcl = bTagCollectorMC.clone(mcPlots=1) #harvest b, c, dusg and ni histos, all not harvested
# module execution
bTagCollectorSequenceMCbcl = cms.Sequence(bTagCollectorMCbcl)
