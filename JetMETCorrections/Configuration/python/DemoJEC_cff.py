import FWCore.ParameterSet.Config as cms
from JetMETCorrections.Configuration.JetCorrectionEra_cff import *
##------------------  DEFINE THE SERVICES  --------------
# L1 (offset) Correction Service
ak5CaloL1 = cms.ESSource(
    'LXXXCorrectionService',
    era       = cms.string(''),
    level     = cms.string('L1Offset'),
    algorithm = cms.string('1PU_IC5Calo'),
    debug     = cms.untracked.bool(True)
    )
# L2 (relative) Correction Service
ak5CaloL2 = cms.ESSource(
    'LXXXCorrectionService',
    JetCorrectionEra,
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5Calo'),
    debug     = cms.untracked.bool(True)
    )
# L3 (absolute) Correction Service
ak5CaloL3 = cms.ESSource(
    'LXXXCorrectionService',
    JetCorrectionEra,
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5Calo'),
    debug     = cms.untracked.bool(True)
    )
# L4 (emf) Correction Service
ak5CaloL4 = cms.ESSource(
    'LXXXCorrectionService',
    era       = cms.string(''),
    level     = cms.string('L4EMF'),
    algorithm = cms.string('AK5Calo'),
    debug     = cms.untracked.bool(True)
    )
# L5 (flavor) Correction Service
ak5CaloL5 = cms.ESSource(
    'LXXXCorrectionService',
    era       = cms.string(''),
    level     = cms.string('L5Flavor'),
    algorithm = cms.string('IC5'),
    section   = cms.untracked.string('bJ'),
    debug     = cms.untracked.bool(True)
    )
# L7 (parton) Correction Service
ak5CaloL7 = cms.ESSource(
    'LXXXCorrectionService',
    era       = cms.string(''),
    level     = cms.string('L7Parton'),
    algorithm = cms.string('AK5'),
    section   = cms.untracked.string('bJ'),
    debug     = cms.untracked.bool(True)
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
