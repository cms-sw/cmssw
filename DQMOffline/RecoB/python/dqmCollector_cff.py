import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.dqmAnalyzer_cff import *

#pf DATA collector
bTagCollectorDATA = bTagHarvest.clone(ptRanges = cms.vdouble(0.0))
# module execution
bTagCollectorSequenceDATA = cms.Sequence(bTagCollectorDATA)

#pf MC collector
bTagCollectorMC = bTagHarvestMC.clone(
    flavPlots = cms.string("allbcl"), #harvest all, b, c, dusg and ni histos 
    ptRanges = cms.vdouble(0.0),
    etaRanges = cms.vdouble(0.0),
)
# module execution
bTagCollectorSequenceMC = cms.Sequence(bTagCollectorMC)
#special sequence for fullsim, all histos havested by the DATA sequence in the dqm offline sequence
bTagCollectorMCbcl = bTagCollectorMC.clone(flavPlots = cms.string("bcl")) #harvest b, c, dusg and ni histos, all not harvested
bTagCollectorSequenceMCbcl = cms.Sequence(bTagCollectorMCbcl)
