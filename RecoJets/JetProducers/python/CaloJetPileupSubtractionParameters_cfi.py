import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: CaloJetPileupSubtractionParameters_cfi.py,v 1.2 2008/04/21 03:28:09 rpw Exp $
CaloJetPileupSubtractionParameters = cms.PSet(
    src = cms.InputTag("towerMaker"),
    inputEtJetMin = cms.double(10.0),
    inputEtMin = cms.double(0.5),
    nSigmaPU = cms.double(1.0),
    radiusPU = cms.double(0.5),
    jetType = cms.untracked.string('CaloJetPileupSubtraction'),
    inputEMin = cms.double(0.0),
    verbose = cms.untracked.bool(False)
)

