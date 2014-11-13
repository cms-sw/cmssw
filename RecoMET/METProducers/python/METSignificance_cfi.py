import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
METSignificance = cms.EDProducer(
    "METSignificanceProducer",
    srcLeptons = cms.VInputTag(
       'slimmedElectrons',
       'slimmedMuons',
       'slimmedPhotons'
       ),
    pfjetsTag            = cms.untracked.InputTag('slimmedJets'),
    metTag               = cms.untracked.InputTag('slimmedMETs'),
    pfcandidatesTag      = cms.untracked.InputTag('packedPFCandidates')
    )
##____________________________________________________________________________||
