################################################################################
#
# JetCorrectionServices_cff
# -------------------------
#
# Define most relevant jet correction services
# for AntiKt R=0.5 CaloJets and PFJets
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
L1Offset = cms.ESSource(
    'LXXXCorrectionService',
    era = cms.string(''),
    level = cms.string('L1Offset'),
    section   = cms.string(''),
    algorithm = cms.string('1PU_IC5Calo'),
    useCondDB = cms.untracked.bool(True)
    )

# L1 (Fastjet PU&UE Subtraction) Correction Service
L1Fastjet = cms.ESSource(
    'L1FastjetCorrectionService',
    era = cms.string(''),
    level       = cms.string(''),
    algorithm   = cms.string('1PU_IC5Calo'),
    section     = cms.string(''),
    srcMedianPt = cms.InputTag('kt6PFJets'),
    useCondDB = cms.untracked.bool(True)
    )

# L2 (relative eta-conformity) Correction Services
ak5CaloL2Relative = cms.ESSource(
    'LXXXCorrectionService',
    era = cms.string(''),
    section   = cms.string(''),
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5Calo'),
    useCondDB = cms.untracked.bool(True)
    )
ak5PFL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5PF' )
#ak5JPTL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5JPT' )
#ak5TrackL2Relative = ak5CaloL2Relative.clone( algorithm = 'AK5TRK' )

# L3 (absolute) Correction Services
ak5CaloL3Absolute = cms.ESSource(
    'LXXXCorrectionService',
    era = cms.string(''),
    section   = cms.string(''),
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5Calo'),
    useCondDB = cms.untracked.bool(True)
    )
ak5PFL3Absolute     = ak5CaloL3Absolute.clone( algorithm = 'AK5PF' )
#ak5JPTL3Absolute    = ak5CaloL3Absolute.clone( algorithm = 'AK5JPT' )
#ak5TrackL3Absolute  = ak5CaloL3Absolute.clone( algorithm = 'AK5TRK' )

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
#ak5JPTL2L3 = cms.ESSource(
#    'JetCorrectionServiceChain',
#    correctors = cms.vstring('ak5JPTL2Relative','ak5JPTL3Absolute')
#    )
#ak5TrackL2L3 = cms.ESSource(
#    'JetCorrectionServiceChain',
#    correctors = cms.vstring('ak5TrackL2Relative','ak5TrackL3Absolute')
#    )

# L1L2L3 CORRECTION SERVICES
ak5CaloL1L2L3 = ak5CaloL2L3.clone()
ak5CaloL1L2L3.correctors.insert(0,'L1Fastjet')
ak5PFL1L2L3 = ak5PFL2L3.clone()
ak5PFL1L2L3.correctors.insert(0,'L1Fastjet')


# L2L3L6 CORRECTION SERVICES
ak5CaloL2L3L6 = ak5CaloL2L3.clone()
ak5CaloL2L3L6.correctors.append('ak5CaloL6SLB')
ak5PFL2L3L6 = ak5PFL2L3.clone()
ak5PFL2L3L6.correctors.append('ak5PFL6SLB')


# L1L2L3L6 CORRECTION SERVICES
ak5CaloL1L2L3L6 = ak5CaloL1L2L3.clone()
ak5CaloL1L2L3L6.correctors.append('ak5CaloL6SLB')
ak5PFL1L2L3L6 = ak5PFL1L2L3.clone()
ak5PFL1L2L3L6.correctors.append('ak5PFL6SLB')
