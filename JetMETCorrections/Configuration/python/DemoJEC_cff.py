import FWCore.ParameterSet.Config as cms
##------------------  DEFINE THE SERVICES  --------------
# L1 (offset) Correction Service
ak5CaloL1 = cms.ESSource(
    'LXXXCorrectionService',
    level     = cms.string('L1Offset'),
    algorithm = cms.string('1PU_IC5Calo'),
    section   = cms.string('')
    )
# L2 (relative) Correction Service
ak5CaloL2 = cms.ESSource(
    'LXXXCorrectionService',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5Calo'),
    section   = cms.string('')
    )
# L3 (absolute) Correction Service
ak5CaloL3 = cms.ESSource(
    'LXXXCorrectionService',
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5Calo'),
    section   = cms.string('')
    )
# L4 (emf) Correction Service
ak5CaloL4 = cms.ESSource(
    'LXXXCorrectionService',
    level     = cms.string('L4EMF'),
    algorithm = cms.string('AK5Calo'),
    section   = cms.string('')
    )
# L5 (flavor) Correction Service
ak5CaloL5 = cms.ESSource(
    'LXXXCorrectionService',
    level     = cms.string('L5Flavor'),
    algorithm = cms.string('IC5Calo'),
    section   = cms.string('bJ')
    )
# L7 (parton) Correction Service
ak5CaloL7 = cms.ESSource(
    'LXXXCorrectionService',
    level     = cms.string('L7Parton'),
    algorithm = cms.string('AK5'),
    section   = cms.untracked.string('bJ')
    )
# Combined Correction Service
ak5CaloL1L2L3L4L5L7 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5CaloL1','ak5CaloL2','ak5CaloL3','ak5CaloL4','ak5CaloL5','ak5CaloL7')
    )

##------------------  DEFINE THE PRODUCER MODULE  ---------
ak5CaloJetsL1L2L3L4L5L7 = cms.EDProducer(
    'CaloJetCorrectionProducer',
    src        = cms.InputTag('ak5CaloJets'),
    correctors = cms.vstring('ak5CaloL1L2L3L4L5L7')
    )
