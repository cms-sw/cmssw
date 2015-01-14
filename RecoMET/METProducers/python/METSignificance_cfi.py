import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.METSignificanceParams_cfi import METSignificanceParams

##____________________________________________________________________________||
METSignificance = cms.EDProducer(
    "METSignificanceProducer",
    srcLeptons = cms.VInputTag(
       'slimmedElectrons',
       'slimmedMuons',
       'slimmedPhotons'
       ),
    srcPfJets            = cms.InputTag('slimmedJets'),
    srcMet               = cms.InputTag('slimmedMETs'),
    srcPFCandidates      = cms.InputTag('packedPFCandidates'),
    
    parameters = METSignificanceParams
    )
##____________________________________________________________________________||
