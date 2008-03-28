import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: CaloJetPileupSubtractionParameters.cfi,v 1.3 2007/06/01 07:37:40 kodolova Exp $
CaloJetPileupSubtractionParameters = cms.PSet(
    src = cms.InputTag("caloTowers"),
    inputEtJetMin = cms.double(10.0),
    inputEtMin = cms.double(0.5),
    nSigmaPU = cms.double(1.0),
    radiusPU = cms.double(0.5),
    jetType = cms.untracked.string('CaloJetPileupSubtraction'),
    inputEMin = cms.double(0.0),
    verbose = cms.untracked.bool(False)
)

