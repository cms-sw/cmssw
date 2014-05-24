import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
# parametrization of MET x/y shift vs. sumEt
from multPhiCorr_dy53X_cfi import multPhiCorr_dy53X as params


pfMEtMultShiftCorr = cms.EDProducer("MultShiftMETcorrInputProducer",
#    src = cms.InputTag('pfMet'), # "raw"/uncorrected PFMEt, needed to access sumEt
    srcPFlow = cms.InputTag('particleFlow', ''),
    parameters = params 
)

pfMEtSysShiftCorrSequence = cms.Sequence( pfMEtMultShiftCorr )
