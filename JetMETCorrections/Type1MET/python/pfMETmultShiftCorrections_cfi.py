import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
# parametrization of MET x/y shift vs. sumEt
from multPhiCorr_phys14_cfi import multPhiCorr_phys14 as params


pfMEtMultShiftCorr = cms.EDProducer("MultShiftMETcorrInputProducer",
    srcPFlow = cms.InputTag('particleFlow', ''),
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    parameters = params 
)

pfMEtSysShiftCorrSequence = cms.Sequence( pfMEtMultShiftCorr )
