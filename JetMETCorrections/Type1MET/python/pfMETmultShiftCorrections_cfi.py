import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------------
# parametrization of MET x/y shift vs. sumEt
from JetMETCorrections.Type1MET.multPhiCorr_741_50nsDY_cfi import multPhiCorr_741_50nsDY as multPhiCorrParams_Txy_50ns
from JetMETCorrections.Type1MET.multPhiCorr_741_25nsDY_cfi import multPhiCorr_741_25nsDY as multPhiCorrParams_Txy_25ns
#from JetMETCorrections.Type1MET.multPhiCorr_763_25nsDY_cfi import multPhiCorr_763_25nsDY as multPhiCorrParams_Txy_25nsTest

#so far, only one set of parameter
# this is ugly, but a direct copy does not work
#50 ns
multPhiCorrParams_T0rtTxy_50ns     = cms.VPSet( pset for pset in multPhiCorrParams_Txy_50ns)
multPhiCorrParams_T0rtT1Txy_50ns   = cms.VPSet( pset for pset in multPhiCorrParams_Txy_50ns)
multPhiCorrParams_T0rtT1T2Txy_50ns = cms.VPSet( pset for pset in multPhiCorrParams_Txy_50ns)
multPhiCorrParams_T0pcTxy_50ns     = cms.VPSet( pset for pset in multPhiCorrParams_Txy_50ns)
multPhiCorrParams_T0pcT1Txy_50ns   = cms.VPSet( pset for pset in multPhiCorrParams_Txy_50ns)
multPhiCorrParams_T0pcT1T2Txy_50ns = cms.VPSet( pset for pset in multPhiCorrParams_Txy_50ns)
multPhiCorrParams_T1Txy_50ns       = cms.VPSet( pset for pset in multPhiCorrParams_Txy_50ns)
multPhiCorrParams_T1T2Txy_50ns     = cms.VPSet( pset for pset in multPhiCorrParams_Txy_50ns)

#25 ns
multPhiCorrParams_T0rtTxy_25ns     = cms.VPSet( pset for pset in multPhiCorrParams_Txy_25ns)
multPhiCorrParams_T0rtT1Txy_25ns   = cms.VPSet( pset for pset in multPhiCorrParams_Txy_25ns)
multPhiCorrParams_T0rtT1T2Txy_25ns = cms.VPSet( pset for pset in multPhiCorrParams_Txy_25ns)
multPhiCorrParams_T0pcTxy_25ns     = cms.VPSet( pset for pset in multPhiCorrParams_Txy_25ns)
multPhiCorrParams_T0pcT1Txy_25ns   = cms.VPSet( pset for pset in multPhiCorrParams_Txy_25ns)
multPhiCorrParams_T0pcT1T2Txy_25ns = cms.VPSet( pset for pset in multPhiCorrParams_Txy_25ns)
multPhiCorrParams_T1Txy_25ns       = cms.VPSet( pset for pset in multPhiCorrParams_Txy_25ns)
multPhiCorrParams_T1T2Txy_25ns     = cms.VPSet( pset for pset in multPhiCorrParams_Txy_25ns)

pfMEtMultShiftCorr = cms.EDProducer("MultShiftMETcorrInputProducer",
    srcPFlow = cms.InputTag('particleFlow', ''),
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    parameters = multPhiCorrParams_Txy_25ns
)
# Run for miniAOD
pfMEtMultShiftCorrDB = cms.EDProducer("MultShiftMETcorrDBInputProducer",
    srcPFlow = cms.InputTag('packedPFCandidates', ''),
    vertexCollection = cms.InputTag('offlineSlimmedPrimaryVertices'),
    isData = cms.untracked.bool(False),
    #sampleType = cms.untracked.string('MC') # MC, Data, DY, TTJets, WJets: MC is default, Data don't need to be specified because of "isData".
    #srcPFlow = cms.InputTag('particleFlow', ''),
    #vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    #parameters = multPhiCorrParams_Txy_25nsTest
)

pfMEtSysShiftCorrSequence = cms.Sequence( pfMEtMultShiftCorr )

#from Configuration.StandardSequences.Eras import eras
#eras.run2_common.toModify(pfMEtMultShiftCorr, parameters=multPhiCorrParams_Txy_25ns )
#eras.run2_50ns_specific.toModify(pfMEtMultShiftCorr, parameters=multPhiCorrParams_Txy_50ns )
#eras.run2_25ns_specific.toModify(pfMEtMultShiftCorr, parameters=multPhiCorrParams_Txy_25ns )


