import FWCore.ParameterSet.Config as cms

hltCaloJetIDProducer = cms.EDProducer("HLTCaloJetIDProducer",
    jetsInput = cms.InputTag("hltMCJetCorJetIcone5HF07"),
    min_EMF = cms.double(0.0001),
    max_EMF = cms.double(999.),
    min_N90 = cms.int32(0),
    min_N90hits = cms.int32(2),
    JetIDParams  = cms.PSet(
         useRecHits      = cms.bool(True),
         hbheRecHitsColl = cms.InputTag("hltHbhereco"),
         hoRecHitsColl   = cms.InputTag("hltHoreco"),
         hfRecHitsColl   = cms.InputTag("hltHfreco"),
         ebRecHitsColl   = cms.InputTag("hltEcalRecHitAll", "EcalRecHitsEB"),
         eeRecHitsColl   = cms.InputTag("hltEcalRecHitAll", "EcalRecHitsEE")
     )                                                      
)


