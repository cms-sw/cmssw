import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: CaloJetParameters_cfi.py,v 1.8 2008/10/21 13:36:15 oehler Exp $
CaloJetParameters = cms.PSet(
    src = cms.InputTag("towerMaker"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(1.0),
    inputEtMin = cms.double(0.5),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0),
    correctInputToSignalVertex = cms.bool(True),
    pvCollection = cms.InputTag('offlinePrimaryVertices'),
    maxBadEcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999)
)

