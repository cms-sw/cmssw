import FWCore.ParameterSet.Config as cms

pfClustersFromCombinedCalo = cms.EDProducer("L1TPFCaloProducer",
     ecalCandidates = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClustersEM'), cms.InputTag('pfClustersFromL1EGClusters')),
     hcalCandidates = cms.VInputTag(),
     hcalDigis = cms.VInputTag(cms.InputTag('simHcalTriggerPrimitiveDigis')),
     hcalHGCTCs = cms.VInputTag(cms.InputTag("hgcalTriggerPrimitiveDigiProducer","calibratedTriggerCells") ),
     hcalHGCTCEtMin = cms.double(0.0),
     emCorrector  = cms.string(""), # no need to correct further
     hadCorrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hadcorr.root"),
     hadCorrectorEmfMax  = cms.double(-1.0),
     ecalClusterer = cms.PSet(
         grid = cms.string("phase1"),
         zsEt = cms.double(0.4),
         seedEt = cms.double(0.5),
         minClusterEt = cms.double(0.5),
         energyWeightedPosition = cms.bool(True),
         energyShareAlgo = cms.string("fractions"),
     ), 
     hcalClusterer = cms.PSet(
         grid = cms.string("phase1"),
         zsEt = cms.double(0.4),
         seedEt = cms.double(0.5),
         minClusterEt = cms.double(0.8),
         energyWeightedPosition = cms.bool(True),
         energyShareAlgo = cms.string("fractions"),
     ),
     linker = cms.PSet(
         grid = cms.string("phase1"),
         hoeCut = cms.double(0.1),
         minPhotonEt = cms.double(1.0),
         minHadronRawEt = cms.double(1.0),
         minHadronEt = cms.double(1.0),
     ),
     resol = cms.PSet(
            etaBins = cms.vdouble( 1.300,  1.700,  2.800,  3.200,  4.000,  5.000),
            offset  = cms.vdouble( 2.644,  1.975,  2.287,  1.113,  0.772,  0.232),
            scale   = cms.vdouble( 0.155,  0.247,  0.130,  0.266,  0.205,  0.302),
            ptMin   = cms.vdouble( 5.000,  5.000,  5.000,  5.000,  5.000,  5.000),
            ptMax   = cms.vdouble(999999, 999999, 999999, 999999, 999999, 999999),
            kind    = cms.string('calo'),
    ),
    debug = cms.untracked.int32(0),
)


