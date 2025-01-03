import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentificationp.muonLinksProducer_cfi import muonLinksProducer
globalMuonLinks = muonLinksProducer.clone(
    inputCollection = cms.InputTag("muons","","@skipCurrentProcess")
)

