import FWCore.ParameterSet.Config as cms

hltEgammaHLTExtraUnseeded = cms.EDProducer("EgammaHLTExtraProducer",
                                           ecalCands = cms.InputTag("hltEgammaCandidatesUnseeded"),
                                           ecal = cms.VPSet(
                                               cms.PSet(
                                                   src= cms.InputTag("hltEcalRecHit","EcalRecHitEB"),
                                                   label = cms.string("EcalRecHitsEB")
                                               ),
                                               cms.PSet(
                                                src= cms.InputTag("hltEcalRecHit","EcalRecHitEE"),
                                                label = cms.string("EcalRecHitsEE")
                                               )
                                           ),
                                           hcal = cms.VPSet(cms.PSet(src=cms.InputTag("hltHbhereco"),label=cms.string(""))),
                                           trks = cms.VPSet(cms.PSet(src=cms.InputTag("hltMergedTracks"),label=cms.string(""))),
                                           pixelSeeds = cms.InputTag("hltEgammaElectronPixelSeedsUnseeded"),
                                           gsfTracks = cms.InputTag("hltEgammaGsfTracksUnseeded"),
                                           minPtToSaveHits = cms.double(8.)
)

