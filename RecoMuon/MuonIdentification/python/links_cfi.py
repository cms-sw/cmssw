import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.muonLinksProducer_cfi import muonLinksProducer as _muonLinksProducer
globalMuonLinks = _muonLinksProducer.clone(
    inputCollection = ("muons","","@skipCurrentProcess")
)

