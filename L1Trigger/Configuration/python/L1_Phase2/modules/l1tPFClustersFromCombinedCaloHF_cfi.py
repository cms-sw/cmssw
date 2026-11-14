import FWCore.ParameterSet.Config as cms

l1tPFClustersFromCombinedCaloHF = cms.EDProducer("L1TPFCaloProducer",
    debug = cms.untracked.int32(0),
    ecalCandidates = cms.VInputTag(),
    ecalClusterer = cms.PSet(
        energyShareAlgo = cms.string('fractions'),
        energyWeightedPosition = cms.bool(True),
        etaBounds = cms.vdouble(-1.0),
        grid = cms.string('phase2'),
        maxClustersEtaPhi = cms.vuint32(),
        minClusterEt = cms.double(0.5),
        phiBounds = cms.vdouble(-1.0),
        seedEt = cms.double(0.5),
        zsEt = cms.double(0.4)
    ),
    emCorrector = cms.string(''),
    hadCorrector = cms.string('L1Trigger/Phase2L1ParticleFlow/data/hfcorr_110X.root'),
    hadCorrectorEmfMax = cms.double(-1.0),
    hcCorrector = cms.string(''),
    hcalCandidates = cms.VInputTag(cms.InputTag("l1tHGCalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering")),
    hcalClusterer = cms.PSet(
        energyShareAlgo = cms.string('fractions'),
        energyWeightedPosition = cms.bool(True),
        etaBounds = cms.vdouble(-1.0),
        grid = cms.string('phase2'),
        maxClustersEtaPhi = cms.vuint32(),
        minClusterEt = cms.double(0.8),
        phiBounds = cms.vdouble(-1.0),
        seedEt = cms.double(0.5),
        zsEt = cms.double(0.4)
    ),
    hcalDigis = cms.VInputTag(cms.InputTag("simHcalTriggerPrimitiveDigis")),
    hcalDigisBarrel = cms.bool(False),
    hcalDigisHF = cms.bool(True),
    hcalHGCTowers = cms.VInputTag(),
    hcalHGCTowersHadOnly = cms.bool(False),
    linker = cms.PSet(
        algo = cms.string('flat'),
        energyShareAlgo = cms.string('fractions'),
        energyWeightedPosition = cms.bool(True),
        etaBounds = cms.vdouble(-1.0),
        grid = cms.string('phase2'),
        hoeCut = cms.double(0.1),
        maxClustersEtaPhi = cms.vuint32(),
        minClusterEt = cms.double(1.0),
        minHadronEt = cms.double(1.0),
        minHadronRawEt = cms.double(1.0),
        minPhotonEt = cms.double(1.0),
        noEmInHGC = cms.bool(False),
        phiBounds = cms.vdouble(-1.0),
        seedEt = cms.double(1.0),
        zsEt = cms.double(0.0)
    ),
    phase2barrelCaloTowers = cms.VInputTag(),
    resol = cms.PSet(
        etaBins = cms.vdouble(3.5, 4.0, 4.5, 5.0),
        kind = cms.string('calo'),
        offset = cms.vdouble(-1.125, 1.22, 1.514, 1.414),
        scale = cms.vdouble(0.868, 0.159, 0.148, 0.194)
    )
)
