import FWCore.ParameterSet.Config as cms

# Standard CalooJets parameters
# $Id: CaloJetParameters_cfi.py,v 1.4 2008/08/20 16:10:09 oehler Exp $
CaloJetParameters = cms.PSet(
    src = cms.InputTag("towerMaker"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(1.0),
    inputEtMin = cms.double(0.5),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

