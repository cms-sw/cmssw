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
ak4CaloL1Offset = cms.ESProducer(
    'L1OffsetCorrectionESProducer',
    level = cms.string('L1Offset'),
    algorithm = cms.string('AK5Calo'),
    vertexCollection = cms.string('offlinePrimaryVertices'),
    minVtxNdof = cms.int32(4)
    )

ak4PFL1Offset = ak4CaloL1Offset.clone(algorithm = 'AK4PF')
ak4PFCHSL1Offset = ak4CaloL1Offset.clone(algorithm = 'AK4PFchs')
ak4JPTL1Offset = ak4CaloL1Offset.clone(algorithm = 'AK5JPT')

# L1 (JPT Offset) Correction Service
ak4L1JPTOffset = cms.ESProducer(
    'L1JPTOffsetCorrectionESProducer',
    level = cms.string('L1JPTOffset'),
    algorithm = cms.string('AK5JPT'),
    offsetService = cms.string('ak4CaloL1Offset')
    )

# L1 (Fastjet PU Subtraction) Correction Service
ak4CaloL1Fastjet = cms.ESProducer(
    'L1FastjetCorrectionESProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK5Calo'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAllCalo'  )
    )
ak4PFL1Fastjet = cms.ESProducer(
    'L1FastjetCorrectionESProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK4PF'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAll' )
    )
ak4PFCHSL1Fastjet = cms.ESProducer(
    'L1FastjetCorrectionESProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK4PFchs'),
    srcRho      = cms.InputTag( 'fixedGridRhoFastjetAll' )
    )
ak4JPTL1Fastjet = ak4CaloL1Fastjet.clone()

# L2 (relative eta-conformity) Correction Services
ak4CaloL2Relative = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5Calo')
    )
ak4PFL2Relative = ak4CaloL2Relative.clone( algorithm = 'AK4PF' )
ak4PFCHSL2Relative = ak4CaloL2Relative.clone( algorithm = 'AK4PFchs' )
ak4JPTL2Relative = ak4CaloL2Relative.clone( algorithm = 'AK5JPT' )
ak4TrackL2Relative = ak4CaloL2Relative.clone( algorithm = 'AK5TRK' )

# L3 (absolute) Correction Services
ak4CaloL3Absolute = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5Calo')
    )
ak4PFL3Absolute     = ak4CaloL3Absolute.clone( algorithm = 'AK4PF' )
ak4PFCHSL3Absolute     = ak4CaloL3Absolute.clone( algorithm = 'AK4PFchs' )
ak4JPTL3Absolute    = ak4CaloL3Absolute.clone( algorithm = 'AK5JPT' )
ak4TrackL3Absolute  = ak4CaloL3Absolute.clone( algorithm = 'AK5TRK' )

# Residual Correction Services
ak4CaloResidual = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2L3Residual'),
    algorithm = cms.string('AK5Calo')
    )
ak4PFResidual  = ak4CaloResidual.clone( algorithm = 'AK4PF' )
ak4PFCHSResidual  = ak4CaloResidual.clone( algorithm = 'AK4PFchs' )
ak4JPTResidual = ak4CaloResidual.clone( algorithm = 'AK5JPT' )

# L6 (semileptonically decaying b-jet) Correction Services
ak4CaloL6SLB = cms.ESProducer(
    'L6SLBCorrectionESProducer',
    level               = cms.string('L6SLB'),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(True),
    srcBTagInfoElectron = cms.InputTag('ak4CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak4CaloJetsSoftMuonTagInfos')
    )
ak4PFL6SLB = cms.ESProducer(
    'L6SLBCorrectionESProducer',
    level               = cms.string('L6SLB'),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(False),
    srcBTagInfoElectron = cms.InputTag('ak4PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak4PFJetsSoftMuonTagInfos')
    )


#
# MULTIPLE LEVEL CORRECTION SERVICES
#

# L2L3 CORRECTION SERVICES
ak4CaloL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL2Relative','ak4CaloL3Absolute')
    )
ak4PFL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL2Relative','ak4PFL3Absolute')
    )
ak4PFCHSL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFCHSL2Relative','ak4PFCHSL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4L1JPTOffset','ak4JPTL2Relative','ak4JPTL3Absolute')
    )
ak4TrackL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4TrackL2Relative','ak4TrackL3Absolute')
    )

# L2L3Residual CORRECTION SERVICES
ak4CaloL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )
ak4PFL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )
ak4PFCHSL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFCHSL2Relative','ak4PFCHSL3Absolute','ak4PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4L1JPTOffset','ak4JPTL2Relative','ak4JPTL3Absolute','ak4JPTResidual')
    )

# L1L2L3 CORRECTION SERVICES
ak4CaloL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL1Offset','ak4CaloL2Relative','ak4CaloL3Absolute')
    )
ak4PFL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL1Offset','ak4PFL2Relative','ak4PFL3Absolute')
    )
ak4PFCHSL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFCHSL1Offset','ak4PFCHSL2Relative','ak4PFCHSL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4L1JPTOffset','ak4JPTL2Relative','ak4JPTL3Absolute')
    )

# L1L2L3Residual CORRECTION SERVICES
ak4CaloL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL1Offset','ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )
ak4PFL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL1Offset','ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )
ak4PFCHSL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFCHSL1Offset','ak4PFCHSL2Relative','ak4PFCHSL3Absolute','ak4PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4L1JPTOffset','ak4JPTL2Relative','ak4JPTL3Absolute','ak4JPTResidual')
    )

# L1L2L3 CORRECTION SERVICES WITH FASTJET
ak4CaloL1FastL2L3 = ak4CaloL2L3.clone()
ak4CaloL1FastL2L3.correctors.insert(0,'ak4CaloL1Fastjet')
ak4PFL1FastL2L3 = ak4PFL2L3.clone()
ak4PFL1FastL2L3.correctors.insert(0,'ak4PFL1Fastjet')
ak4PFCHSL1FastL2L3 = ak4PFCHSL2L3.clone()
ak4PFCHSL1FastL2L3.correctors.insert(0,'ak4PFCHSL1Fastjet')
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1FastL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4JPTL1Fastjet','ak4JPTL2Relative','ak4JPTL3Absolute')
    )

# L1L2L3Residual CORRECTION SERVICES WITH FASTJET
ak4CaloL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4CaloL1Fastjet','ak4CaloL2Relative','ak4CaloL3Absolute','ak4CaloResidual')
    )
ak4PFL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFL1Fastjet','ak4PFL2Relative','ak4PFL3Absolute','ak4PFResidual')
    )
ak4PFCHSL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFCHSL1Fastjet','ak4PFCHSL2Relative','ak4PFCHSL3Absolute','ak4PFCHSResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak4JPTL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4JPTL1Fastjet','ak4JPTL2Relative','ak4JPTL3Absolute','ak4JPTResidual')
    )

# L2L3L6 CORRECTION SERVICES
ak4CaloL2L3L6 = ak4CaloL2L3.clone()
ak4CaloL2L3L6.correctors.append('ak4CaloL6SLB')
ak4PFL2L3L6 = ak4PFL2L3.clone()
ak4PFL2L3L6.correctors.append('ak4PFL6SLB')

# L1L2L3L6 CORRECTION SERVICES
ak4CaloL1FastL2L3L6 = ak4CaloL1FastL2L3.clone()
ak4CaloL1FastL2L3L6.correctors.append('ak4CaloL6SLB')
ak4PFL1FastL2L3L6 = ak4PFL1FastL2L3.clone()
ak4PFL1FastL2L3L6.correctors.append('ak4PFL6SLB')
