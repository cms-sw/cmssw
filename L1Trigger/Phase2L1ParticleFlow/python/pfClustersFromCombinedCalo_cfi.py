import FWCore.ParameterSet.Config as cms

pfClustersFromCombinedCalo = cms.EDProducer("L1TPFCaloProducer",
     ecalCandidates = cms.VInputTag(cms.InputTag('pfClustersFromL1EGClusters')), # using EM from towers in HGC, no longer reading also 'pfClustersFromHGC3DClustersEM'  
     hcalCandidates = cms.VInputTag(),
     hcalDigis = cms.VInputTag(cms.InputTag('simHcalTriggerPrimitiveDigis')),
     hcalHGCTCs = cms.VInputTag(), #cms.InputTag("hgcalTriggerPrimitiveDigiProducer","calibratedTriggerCells") ),
     hcalHGCTowers = cms.VInputTag(cms.InputTag("hgcalTriggerPrimitiveDigiProducer","tower") ),
     hcalHGCTowersHadOnly = cms.bool(False), # take also EM part from towers
     hcalHGCTCEtMin = cms.double(0.0),
     emCorrector  = cms.string(""), # no need to correct further
     hcCorrector  = cms.string(""), # no correction to hcal-only in the default scheme
     hadCorrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hadcorr.root"), # correction on linked cluster
     hadCorrectorEmfMax  = cms.double(-1.0),
     ecalClusterer = cms.PSet(
         grid = cms.string("phase2"),
         zsEt = cms.double(0.4),
         seedEt = cms.double(0.5),
         minClusterEt = cms.double(0.5),
         energyWeightedPosition = cms.bool(True),
         energyShareAlgo = cms.string("fractions"),
     ), 
     hcalClusterer = cms.PSet(
         grid = cms.string("phase2"),
         zsEt = cms.double(0.4),
         seedEt = cms.double(0.5),
         minClusterEt = cms.double(0.8),
         energyWeightedPosition = cms.bool(True),
         energyShareAlgo = cms.string("fractions"),
     ),
     linker = cms.PSet(
         algo = cms.string("flat"),

         zsEt = cms.double(0.0), ## Ecal and Hcal are already ZS-ed above
         seedEt = cms.double(1.0),
         minClusterEt = cms.double(1.0),
         energyWeightedPosition = cms.bool(True),
         energyShareAlgo = cms.string("fractions"),
 
         grid = cms.string("phase2"),
         hoeCut = cms.double(0.1),
         minPhotonEt = cms.double(1.0),
         minHadronRawEt = cms.double(1.0),
         minHadronEt = cms.double(1.0),
     ),
     resol = cms.PSet(
            etaBins = cms.vdouble( 1.300,  1.700,  2.800,  3.200,  4.000,  5.000),
            offset  = cms.vdouble( 2.688,  1.382,  2.096,  1.022,  0.757,  0.185),
            scale   = cms.vdouble( 0.154,  0.341,  0.105,  0.255,  0.208,  0.306),
            ptMin   = cms.vdouble( 5.000,  5.000,  5.000,  5.000,  5.000,  5.000),
            ptMax   = cms.vdouble(999999, 999999, 999999, 999999, 999999, 999999),
            kind    = cms.string('calo'),
    ),
    debug = cms.untracked.int32(0),
)


