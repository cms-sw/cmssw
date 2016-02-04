import FWCore.ParameterSet.Config as cms

ecalRecHitMerger = cms.EDProducer("EcalRecHitsMerger",
    EgammaSource_EB = cms.untracked.InputTag("ecalRegionalEgammaRecHitTmp","EcalRecHitsEB"),
    MuonsSource_EB = cms.untracked.InputTag("ecalRegionalMuonsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EB = cms.untracked.InputTag("ecalRegionalJetsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EE = cms.untracked.InputTag("ecalRegionalJetsRecHitTmp","EcalRecHitsEE"),
    MuonsSource_EE = cms.untracked.InputTag("ecalRegionalMuonsRecHitTmp","EcalRecHitsEE"),
    EcalRecHitCollectionEB = cms.untracked.string('EcalRecHitsEB'),
    RestSource_EE = cms.untracked.InputTag("ecalRegionalRestRecHitTmp","EcalRecHitsEE"),
    RestSource_EB = cms.untracked.InputTag("ecalRegionalRestRecHitTmp","EcalRecHitsEB"),
    TausSource_EB = cms.untracked.InputTag("ecalRegionalTausRecHitTmp","EcalRecHitsEB"),
    TausSource_EE = cms.untracked.InputTag("ecalRegionalTausRecHitTmp","EcalRecHitsEE"),
    debug = cms.untracked.bool(False),
    EcalRecHitCollectionEE = cms.untracked.string('EcalRecHitsEE'),
    OutputLabel_EE = cms.untracked.string('EcalRecHitsEE'),
    EgammaSource_EE = cms.untracked.InputTag("ecalRegionalEgammaRecHitTmp","EcalRecHitsEE"),
    OutputLabel_EB = cms.untracked.string('EcalRecHitsEB')
)


