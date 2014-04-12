################################################################################
#
# JetCorrectionServices_cff
# -------------------------
#
# Define most relevant jet correction services
# for AntiKt R=0.5 CaloJets, PFJets, JPTJets and TrackJets
#
################################################################################
import FWCore.ParameterSet.Config as cms

# This is now pulled from the Global Tag. For testing purposes, you can
# continue to use this file but it is no longer on by default.
#from JetMETCorrections.Configuration.JetCorrectionCondDB_cff import *


#
# SINGLE LEVEL CORRECTION SERVICES
#

# L1 (Offset) Correction Service
ak5CaloL1Offset = cms.ESProducer(
    'L1OffsetCorrectionESProducer',
    level = cms.string('L1Offset'),
    algorithm = cms.string('AK5Calo'),
    vertexCollection = cms.string('offlinePrimaryVertices'),
    minVtxNdof = cms.int32(4)
    )

ak5PFL1Offset = ak5CaloL1Offset.clone(algorithm = 'AK5PF')
ak5PFCHSL1Offset = ak5CaloL1Offset.clone(algorithm = 'AK5PFCHS') 
ak5JPTL1Offset = ak5CaloL1Offset.clone(algorithm = 'AK5JPT')

# L1 (JPT Offset) Correction Service
ak5L1JPTOffset = cms.ESProducer(
    'L1JPTOffsetCorrectionESProducer',
    level = cms.string('L1JPTOffset'),
    algorithm = cms.string('AK5JPT'),
    offsetService = cms.string('ak5CaloL1Offset')
    )

# L1 (Fastjet PU Subtraction) Correction Service
ak5CaloL1Fastjet = cms.ESProducer(
    'L1FastjetCorrectionESProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK5Calo'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAllCalo'  )
    )
ak5PFL1Fastjet = cms.ESProducer(
    'L1FastjetCorrectionESProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK5PF'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAll' )
    )
ak5PFCHSL1Fastjet = cms.ESProducer(
    'L1FastjetCorrectionESProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK5PFCHS'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAll' )
    )
ak5JPTL1Fastjet = ak5CaloL1Fastjet.clone()

# L2 (relative eta-conformity) Correction Services
ak5CaloL2Relative = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5Calo')
    )
ak5PFL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5PF' )
ak5PFCHSL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5PFCHS' )
ak5JPTL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5JPT' )
ak5TrackL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5TRK' )

# L3 (absolute) Correction Services
ak5CaloL3Absolute = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5Calo')
    )
ak5PFL3Absolute     = ak5CaloL3Absolute.clone( algorithm = 'AK5PF' )
ak5PFCHSL3Absolute     = ak5CaloL3Absolute.clone( algorithm = 'AK5PFCHS' )
ak5JPTL3Absolute    = ak5CaloL3Absolute.clone( algorithm = 'AK5JPT' )
ak5TrackL3Absolute  = ak5CaloL3Absolute.clone( algorithm = 'AK5TRK' )

# Residual Correction Services
ak5CaloResidual = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2L3Residual'),
    algorithm = cms.string('AK5Calo')
    )
ak5PFResidual  = ak5CaloResidual.clone( algorithm = 'AK5PF' )
ak5PFCHSResidual  = ak5CaloResidual.clone( algorithm = 'AK5PFCHS' )
ak5JPTResidual = ak5CaloResidual.clone( algorithm = 'AK5JPT' )

# L6 (semileptonically decaying b-jet) Correction Services
ak5CaloL6SLB = cms.ESProducer(
    'L6SLBCorrectionESProducer',
    level               = cms.string('L6SLB'),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(True),
    srcBTagInfoElectron = cms.InputTag('ak5CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak5CaloJetsSoftMuonTagInfos')
    )
ak5PFL6SLB = cms.ESProducer(
    'L6SLBCorrectionESProducer',
    level               = cms.string('L6SLB'),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(False),
    srcBTagInfoElectron = cms.InputTag('ak5PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak5PFJetsSoftMuonTagInfos')
    )


#
# MULTIPLE LEVEL CORRECTION SERVICES
#

# L2L3 CORRECTION SERVICES
ak5CaloL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5CaloL2Relative','ak5CaloL3Absolute')
    )
ak5PFL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFL2Relative','ak5PFL3Absolute')
    )
ak5PFCHSL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFCHSL2Relative','ak5PFCHSL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute')
    )
ak5TrackL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5TrackL2Relative','ak5TrackL3Absolute')
    )

# L2L3Residual CORRECTION SERVICES
ak5CaloL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5CaloL2Relative','ak5CaloL3Absolute','ak5CaloResidual')
    )
ak5PFL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFL2Relative','ak5PFL3Absolute','ak5PFResidual')
    )
ak5PFCHSL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFCHSL2Relative','ak5PFCHSL3Absolute','ak5PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute','ak5JPTResidual')
    )

# L1L2L3 CORRECTION SERVICES
ak5CaloL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5CaloL1Offset','ak5CaloL2Relative','ak5CaloL3Absolute')
    )
ak5PFL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFL1Offset','ak5PFL2Relative','ak5PFL3Absolute')
    )
ak5PFCHSL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFCHSL1Offset','ak5PFCHSL2Relative','ak5PFCHSL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute')
    )

# L1L2L3Residual CORRECTION SERVICES
ak5CaloL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5CaloL1Offset','ak5CaloL2Relative','ak5CaloL3Absolute','ak5CaloResidual')
    )
ak5PFL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFL1Offset','ak5PFL2Relative','ak5PFL3Absolute','ak5PFResidual')
    )
ak5PFCHSL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFCHSL1Offset','ak5PFCHSL2Relative','ak5PFCHSL3Absolute','ak5PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute','ak5JPTResidual')
    )

# L1L2L3 CORRECTION SERVICES WITH FASTJET
ak5CaloL1FastL2L3 = ak5CaloL2L3.clone()
ak5CaloL1FastL2L3.correctors.insert(0,'ak5CaloL1Fastjet')
ak5PFL1FastL2L3 = ak5PFL2L3.clone()
ak5PFL1FastL2L3.correctors.insert(0,'ak5PFL1Fastjet')
ak5PFCHSL1FastL2L3 = ak5PFCHSL2L3.clone()
ak5PFCHSL1FastL2L3.correctors.insert(0,'ak5PFCHSL1Fastjet')
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL1FastL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5JPTL1Fastjet','ak5JPTL2Relative','ak5JPTL3Absolute')
    )

# L1L2L3Residual CORRECTION SERVICES WITH FASTJET
ak5CaloL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5CaloL1Fastjet','ak5CaloL2Relative','ak5CaloL3Absolute','ak5CaloResidual')
    )
ak5PFL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFL1Fastjet','ak5PFL2Relative','ak5PFL3Absolute','ak5PFResidual')
    )
ak5PFCHSL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFCHSL1Fastjet','ak5PFCHSL2Relative','ak5PFCHSL3Absolute','ak5PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5JPTL1Fastjet','ak5JPTL2Relative','ak5JPTL3Absolute','ak5JPTResidual')
    )

# L2L3L6 CORRECTION SERVICES
ak5CaloL2L3L6 = ak5CaloL2L3.clone()
ak5CaloL2L3L6.correctors.append('ak5CaloL6SLB')
ak5PFL2L3L6 = ak5PFL2L3.clone()
ak5PFL2L3L6.correctors.append('ak5PFL6SLB')

# L1L2L3L6 CORRECTION SERVICES
ak5CaloL1FastL2L3L6 = ak5CaloL1FastL2L3.clone()
ak5CaloL1FastL2L3L6.correctors.append('ak5CaloL6SLB')
ak5PFL1FastL2L3L6 = ak5PFL1FastL2L3.clone()
ak5PFL1FastL2L3L6.correctors.append('ak5PFL6SLB')
