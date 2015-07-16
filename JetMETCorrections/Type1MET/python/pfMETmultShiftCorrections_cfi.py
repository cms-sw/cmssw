import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
# parametrization of MET x/y shift vs. sumEt
from JetMETCorrections.Type1MET.multPhiCorr_74X_50ns_cfi import multPhiCorr_741 as multPhiCorrParams_Txy

#so far, only one set of parameter
# this is ugly, but a direct copy does not work
multPhiCorrParams_T0rtTxy     = cms.VPSet( pset for pset in multPhiCorrParams_Txy)
multPhiCorrParams_T0rtT1Txy   = cms.VPSet( pset for pset in multPhiCorrParams_Txy)
multPhiCorrParams_T0rtT1T2Txy = cms.VPSet( pset for pset in multPhiCorrParams_Txy)
multPhiCorrParams_T0pcTxy     = cms.VPSet( pset for pset in multPhiCorrParams_Txy)
multPhiCorrParams_T0pcT1Txy   = cms.VPSet( pset for pset in multPhiCorrParams_Txy)
multPhiCorrParams_T0pcT1T2Txy = cms.VPSet( pset for pset in multPhiCorrParams_Txy)
multPhiCorrParams_T1Txy       = cms.VPSet( pset for pset in multPhiCorrParams_Txy)
multPhiCorrParams_T1T2Txy     = cms.VPSet( pset for pset in multPhiCorrParams_Txy)


pfMEtMultShiftCorr = cms.EDProducer("MultShiftMETcorrInputProducer",
    srcPFlow = cms.InputTag('particleFlow', ''),
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    parameters = multPhiCorrParams_Txy
)

pfMEtSysShiftCorrSequence = cms.Sequence( pfMEtMultShiftCorr )

