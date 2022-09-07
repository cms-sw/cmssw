import FWCore.ParameterSet.Config as cms

l1tPFClustersFromCombinedCalo = cms.EDProducer("L1TPFCaloProducer",
     ecalCandidates = cms.VInputTag(cms.InputTag('l1tPFClustersFromL1EGClusters:all')), 
     hcalCandidates = cms.VInputTag(),
     hcalDigis = cms.VInputTag(cms.InputTag('simHcalTriggerPrimitiveDigis')),
     hcalDigisBarrel = cms.bool(False),
     hcalDigisHF = cms.bool(True),
     phase2barrelCaloTowers = cms.VInputTag(cms.InputTag("l1tEGammaClusterEmuProducer","L1CaloTowerCollection","")),
     hcalHGCTowers = cms.VInputTag(cms.InputTag("l1tHGCalTowerProducer:HGCalTowerProcessor") ),
     hcalHGCTowersHadOnly = cms.bool(False), # take also EM part from towers
     emCorrector  = cms.string(""), # no need to correct further
     hcCorrector  = cms.string(""), # no correction to hcal-only in the default scheme
     hadCorrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hadcorr.root"), # correction on linked cluster
     hadCorrectorEmfMax  = cms.double(-1.0),
     ecalClusterer = cms.PSet(
         grid = cms.string("phase2"),
         zsEt = cms.double(0.4),
         seedEt = cms.double(0.5),
         etaBounds = cms.vdouble(-1.),
         phiBounds = cms.vdouble(-1.),
         maxClustersEtaPhi = cms.vuint32(),
         minClusterEt = cms.double(0.5),
         energyWeightedPosition = cms.bool(True),
         energyShareAlgo = cms.string("fractions"),
     ), 
     hcalClusterer = cms.PSet(
         grid = cms.string("phase2"),
         zsEt = cms.double(0.4),
         seedEt = cms.double(0.5),
         etaBounds = cms.vdouble(-1.),
         phiBounds = cms.vdouble(-1.),
         maxClustersEtaPhi = cms.vuint32(),
         minClusterEt = cms.double(0.8),
         energyWeightedPosition = cms.bool(True),
         energyShareAlgo = cms.string("fractions"),
     ),
     linker = cms.PSet(
         algo = cms.string("flat"),

         zsEt = cms.double(0.0), ## Ecal and Hcal are already ZS-ed above
         seedEt = cms.double(1.0),
         etaBounds = cms.vdouble(-1.),
         phiBounds = cms.vdouble(-1.),
         maxClustersEtaPhi = cms.vuint32(),
         minClusterEt = cms.double(1.0),
         energyWeightedPosition = cms.bool(True),
         energyShareAlgo = cms.string("fractions"),
 
         grid = cms.string("phase2"),
         hoeCut = cms.double(0.1),
         minPhotonEt = cms.double(1.0),
         minHadronRawEt = cms.double(1.0),
         minHadronEt = cms.double(1.0),
         noEmInHGC = cms.bool(False)
     ),
     resol = cms.PSet(
            etaBins = cms.vdouble( 1.300,  1.700,  2.800,  3.200,  4.000,  5.000),
            offset  = cms.vdouble( 2.572,  1.759,  1.858,  2.407,  1.185,  1.658),
            scale   = cms.vdouble( 0.132,  0.240,  0.090,  0.138,  0.143,  0.147),
            kind    = cms.string('calo'),
    ),
    debug = cms.untracked.int32(0),
)


