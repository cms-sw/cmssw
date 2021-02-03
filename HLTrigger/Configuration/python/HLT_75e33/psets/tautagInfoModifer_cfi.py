import FWCore.ParameterSet.Config as cms

tautagInfoModifer = cms.PSet(
    name = cms.string('TTIworkaround'),
    pfTauTagInfoSrc = cms.InputTag("pfRecoTauTagInfoProducer"),
    plugin = cms.string('RecoTauTagInfoWorkaroundModifer')
)