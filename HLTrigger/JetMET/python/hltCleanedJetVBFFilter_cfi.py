import FWCore.ParameterSet.Config as cms

hltJetVBFFilter = cms.EDFilter("HLTCleanedJetVBFFilter",
    inputJetTag = cms.InputTag("myJets"),
    inputEleTag = cms.InputTag("myElectrons"),
    saveTag     = cms.untracked.bool( False ),
    minCleaningDR = cms.double(1.0),
    minJetEtHigh  = cms.double(1.0),
    minJetEtLow   = cms.double(1.0),
    minJetDeta    = cms.double(1.0)
)
