import FWCore.ParameterSet.Config as cms

hltEgammaHLTExtra = cms.EDProducer("EgammaHLTExtraProducer",
                                   egCands = cms.VPSet(
                                       cms.PSet(
                                           pixelSeeds = cms.InputTag("hltEgammaElectronPixelSeeds"),
                                           ecalCands = cms.InputTag("hltEgammaCandidates"),
                                           gsfTracks = cms.InputTag("hltEgammaGsfTracks"),
                                           label = cms.string('')
                                       ),
                                       cms.PSet(
                                           pixelSeeds = cms.InputTag("hltEgammaElectronPixelSeedsUnseeded"),
                                           ecalCands = cms.InputTag("hltEgammaCandidatesUnseeded"),
                                           gsfTracks = cms.InputTag("hltEgammaGsfTracksUnseeded"),
                                           label = cms.string('Unseeded')
                                       ),
                                   ),                 
                                   ecal = cms.VPSet(
                                       cms.PSet(
                                           src= cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
                                           label = cms.string("EcalRecHitsEB")
                                       ),
                                       cms.PSet(
                                           src= cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
                                           label = cms.string("EcalRecHitsEE")
                                       )
                                   ),
                                   pfClusIso = cms.VPSet(
                                       cms.PSet(
                                           src = cms.InputTag("hltParticleFlowClusterECALL1Seeded"),
                                           label = cms.string("Ecal")
                                       ),
                                       cms.PSet(
                                           src = cms.InputTag("hltParticleFlowClusterECALUnseeded"),
                                           label = cms.string("EcalUnseeded")
                                       ),
                                       cms.PSet(
                                           src = cms.InputTag("hltParticleFlowClusterHCAL"),
                                           label = cms.string("Hcal")
                                       ),
                                   ),
                                   hcal = cms.VPSet(cms.PSet(src=cms.InputTag("hltHbhereco"),label=cms.string(""))),
                                   trks = cms.VPSet(cms.PSet(src=cms.InputTag("hltMergedTracks"),label=cms.string(""))),                                  
                                   minPtToSaveHits = cms.double(8.),
                                   saveHitsPlusHalfPi = cms.bool(True),
                                   saveHitsPlusPi = cms.bool(False)
                                   
)

