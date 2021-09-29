import FWCore.ParameterSet.Config as cms

pfClustersFromCombinedCaloHCal = cms.EDProducer("L1TPFCaloProducer",
    debug = cms.untracked.int32(0),
    ecalCandidates = cms.VInputTag(cms.InputTag("pfClustersFromL1EGClusters")),
    ecalClusterer = cms.PSet(
        energyShareAlgo = cms.string('fractions'),
        energyWeightedPosition = cms.bool(True),
        grid = cms.string('phase2'),
        minClusterEt = cms.double(0.5),
        seedEt = cms.double(0.5),
        zsEt = cms.double(0.4)
    ),
    emCorrector = cms.string(''),
    hadCorrector = cms.string('L1Trigger/Phase2L1ParticleFlow/data/hadcorr_barrel_110X.root'),
    hadCorrectorEmfMax = cms.double(-1.0),
    hcCorrector = cms.string(''),
    hcalCandidates = cms.VInputTag(),
    hcalClusterer = cms.PSet(
        energyShareAlgo = cms.string('fractions'),
        energyWeightedPosition = cms.bool(True),
        grid = cms.string('phase2'),
        minClusterEt = cms.double(0.8),
        seedEt = cms.double(0.5),
        zsEt = cms.double(0.4)
    ),
    hcalDigis = cms.VInputTag(),
    hcalDigisBarrel = cms.bool(True),
    hcalDigisHF = cms.bool(False),
    hcalHGCTowers = cms.VInputTag(),
    hcalHGCTowersHadOnly = cms.bool(False),
    linker = cms.PSet(
        algo = cms.string('flat'),
        energyShareAlgo = cms.string('fractions'),
        energyWeightedPosition = cms.bool(True),
        grid = cms.string('phase2'),
        hoeCut = cms.double(0.1),
        minClusterEt = cms.double(1.0),
        minHadronEt = cms.double(1.0),
        minHadronRawEt = cms.double(1.0),
        minPhotonEt = cms.double(1.0),
        noEmInHGC = cms.bool(False),
        seedEt = cms.double(1.0),
        zsEt = cms.double(0.0)
    ),
    phase2barrelCaloTowers = cms.VInputTag(cms.InputTag("L1EGammaClusterEmuProducer")),
    resol = cms.PSet(
        etaBins = cms.vdouble(0.7, 1.2, 1.6),
        kind = cms.string('calo'),
        offset = cms.vdouble(2.909, 2.864, 0.294),
        scale = cms.vdouble(0.119, 0.127, 0.442)
    )
)
