import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: CaloJetParameters_cfi.py,v 1.5 2008/09/20 17:49:54 oehler Exp $
CaloJetParameters = cms.PSet(
    src = cms.InputTag("towerMaker"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(1.0),
    inputEtMin = cms.double(0.5),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0),
    correctInputToSignalVertex = cms.bool(True),
    pvCollection = cms.InputTag('offlinePrimaryVertices')
)

