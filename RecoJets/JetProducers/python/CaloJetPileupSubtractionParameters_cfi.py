import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: CaloJetPileupSubtractionParameters_cfi.py,v 1.3 2008/07/16 15:01:07 kodolova Exp $
CaloJetPileupSubtractionParameters = cms.PSet(
    src = cms.InputTag("towerMaker"),
    inputEtJetMin = cms.double(10.0),
    inputEtMin = cms.double(0.5),
    nSigmaPU = cms.double(1.0),
    radiusPU = cms.double(0.5),
    jetType = cms.untracked.string('CaloJetPileupSubtraction'),
    inputEMin = cms.double(0.0),
    verbose = cms.untracked.bool(False),
    maxBadEcalCells = cms.uint32(9999999),
    maxRecoveredEcalCells = cms.uint32(9999999),
    maxProblematicEcalCells = cms.uint32(9999999),
    maxBadHcalCells = cms.uint32(9999999),
    maxRecoveredHcalCells = cms.uint32(9999999),
    maxProblematicHcalCells = cms.uint32(9999999)
)

