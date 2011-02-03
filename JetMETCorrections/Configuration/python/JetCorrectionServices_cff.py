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
ak5CaloL1Offset = cms.ESSource(
    'L1OffsetCorrectionService',
    era = cms.string('Fall10'),
    level = cms.string('L1Offset'),
    section   = cms.string(''),
    algorithm = cms.string('AK5Calo'),
    useCondDB = cms.untracked.bool(True),
    vertexCollection = cms.string('offlinePrimaryVertices'),
    minVtxNdof = cms.int32(4)
    )

ak5PFL1Offset = ak5CaloL1Offset.clone(algorithm = 'AK5PF') 
ak5JPTL1Offset = ak5CaloL1Offset.clone()

# L1 (JPT Offset) Correction Service
ak5L1JPTOffset = cms.ESSource(
    'LXXXCorrectionService',
    era = cms.string('Summer10'),
    level = cms.string('L1JPTOffset'),
    section   = cms.string(''),
    algorithm = cms.string('AK5JPT')
    )

# L1 (Fastjet PU Subtraction) Correction Service
ak5CaloL1Fastjet = cms.ESSource(
    'L1FastjetCorrectionService',
    era         = cms.string('Jec10V1'),
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK5Calo'),
    section     = cms.string(''),
    srcRho      = cms.InputTag('kt6PFJets','rho'),
    useCondDB = cms.untracked.bool(True)
    )
ak5PFL1Fastjet = ak5CaloL1Fastjet.clone(algorithm = 'AK5PF')
ak5JPTL1Fastjet = ak5CaloL1Fastjet.clone()

# L2 (relative eta-conformity) Correction Services
ak5CaloL2Relative = cms.ESSource(
    'LXXXCorrectionService',
    era = cms.string('Spring10'),
    section   = cms.string(''),
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5Calo'),
    useCondDB = cms.untracked.bool(True)
    )
ak5PFL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5PF' )
ak5JPTL2Relative = ak5CaloL2Relative.clone( era = 'Summer10', algorithm = 'AK5JPT' )
ak5TrackL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5TRK' )

# L3 (absolute) Correction Services
ak5CaloL3Absolute = cms.ESSource(
    'LXXXCorrectionService',
    era = cms.string('Spring10'),
    section   = cms.string(''),
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5Calo'),
    useCondDB = cms.untracked.bool(True)
    )
ak5PFL3Absolute     = ak5CaloL3Absolute.clone( algorithm = 'AK5PF' )
ak5JPTL3Absolute    = ak5CaloL3Absolute.clone( era = 'Summer10', algorithm = 'AK5JPT' )
ak5TrackL3Absolute  = ak5CaloL3Absolute.clone( algorithm = 'AK5TRK' )

# Residual Correction Services
ak5CaloResidual = cms.ESSource(
    'LXXXCorrectionService',
    era = cms.string('Spring10DataV2'),
    section   = cms.string(''),
    level     = cms.string('L2L3Residual'),
    algorithm = cms.string('AK5Calo'),
    useCondDB = cms.untracked.bool(True)
    )
ak5PFResidual  = ak5CaloResidual.clone( algorithm = 'AK5PF' )
ak5JPTResidual = ak5CaloResidual.clone( algorithm = 'AK5JPT' )

# L6 (semileptonically decaying b-jet) Correction Services
ak5CaloL6SLB = cms.ESSource(
    'L6SLBCorrectionService',
    era                 = cms.string(''),
    level               = cms.string('L6SLB'),
    section             = cms.string(''),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(True),
    srcBTagInfoElectron = cms.InputTag('ak5CaloJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak5CaloJetsSoftMuonTagInfos'),
    useCondDB = cms.untracked.bool(True)
    )
ak5PFL6SLB = cms.ESSource(
    'L6SLBCorrectionService',
    era                 = cms.string(''),
    level               = cms.string('L6SLB'),
    section             = cms.string(''),
    algorithm           = cms.string(''),
    addMuonToJet        = cms.bool(False),
    srcBTagInfoElectron = cms.InputTag('ak5PFJetsSoftElectronTagInfos'),
    srcBTagInfoMuon     = cms.InputTag('ak5PFJetsSoftMuonTagInfos'),
    useCondDB = cms.untracked.bool(True)
    )


#
# MULTIPLE LEVEL CORRECTION SERVICES
#

# L2L3 CORRECTION SERVICES
ak5CaloL2L3 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5CaloL2Relative','ak5CaloL3Absolute')
    )
ak5PFL2L3 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5PFL2Relative','ak5PFL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL2L3 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute')
    )
ak5TrackL2L3 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5TrackL2Relative','ak5TrackL3Absolute')
    )

# L2L3Residual CORRECTION SERVICES
ak5CaloL2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5CaloL2Relative','ak5CaloL3Absolute','ak5CaloResidual')
    )
ak5PFL2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5PFL2Relative','ak5PFL3Absolute','ak5PFResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute','ak5JPTResidual')
    )

# L1L2L3 CORRECTION SERVICES
ak5CaloL1L2L3 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5CaloL1Offset','ak5CaloL2Relative','ak5CaloL3Absolute')
    )
ak5PFL1L2L3 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5PFL1Offset','ak5PFL2Relative','ak5PFL3Absolute')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL1L2L3 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5JPTL1Offset','ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute')
    )

# L1L2L3Residual CORRECTION SERVICES
ak5CaloL1L2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5CaloL1Offset','ak5CaloL2Relative','ak5CaloL3Absolute','ak5CaloResidual')
    )
ak5PFL1L2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5PFL1Offset','ak5PFL2Relative','ak5PFL3Absolute','ak5PFResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL1L2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5JPTL1Offset','ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute','ak5JPTResidual')
    )

# L1L2L3 CORRECTION SERVICES WITH FASTJET
ak5CaloL1FastL2L3 = ak5CaloL2L3.clone()
ak5CaloL1FastL2L3.correctors.insert(0,'ak5CaloL1Fastjet')
ak5PFL1FastL2L3 = ak5PFL2L3.clone()
ak5PFL1FastL2L3.correctors.insert(0,'ak5PFL1Fastjet')
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL1FastL2L3 = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5JPTL1Fastjet','ak5JPTL1Offset','ak5JPTL2Relative','ak5JPTL3Absolute')
    )

# L1L2L3Residual CORRECTION SERVICES WITH FASTJET
ak5CaloL1FastL2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5CaloL1Fastjet','ak5CaloL2Relative','ak5CaloL3Absolute','ak5CaloResidual')
    )
ak5PFL1FastL2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5PFL1Fastjet','ak5PFL2Relative','ak5PFL3Absolute','ak5PFResidual')
    )
#--- JPT needs the L1JPTOffset to account for the ZSP changes.
#--- L1JPTOffset is NOT the same as L1Offset !!!!!
ak5JPTL1FastL2L3Residual = cms.ESSource(
    'JetCorrectionServiceChain',
    correctors = cms.vstring('ak5JPTL1Fastjet','ak5L1JPTOffset','ak5JPTL2Relative','ak5JPTL3Absolute','ak5JPTResidual')
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
