import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.alpaka_cff import alpaka

def customise_alpaka_seeds(process):

    process.hltEgammaElectronPixelSeedsPortable = cms.EDProducer("ElectronNHitSeedAlpakaProducer@alpaka",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        initialSeeds = cms.InputTag("hltElePixelSeedsCombinedL1Seeded"),
        superClusters = cms.InputTag("hltEgammaSuperClustersToPixelMatchL1Seeded")
    )

    for taskName, task in process.tasks_().items():
        if task.contains(process.hltEgammaElectronPixelSeedsL1Seeded):
            task.add(process.hltEgammaElectronPixelSeedsPortable)

    for seqName, seq in process.sequences_().items():
        if seq.contains(process.hltEgammaElectronPixelSeedsL1Seeded):
            seq.replace(process.hltEgammaElectronPixelSeedsL1Seeded,
                        process.hltEgammaElectronPixelSeedsPortable + process.hltEgammaElectronPixelSeedsL1Seeded)

    alpaka_converter = cms.EDProducer("ElectronSeedConverter",
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        detLayerGeom = cms.ESInputTag("","GlobalDetLayerGeometry"),
        eleSeedsSoA = cms.InputTag("hltEgammaElectronPixelSeedsPortable"),
        initialSeeds = cms.InputTag("hltElePixelSeedsCombinedL1Seeded"),
        measTkEvt = cms.InputTag("hltMeasurementTrackerEvent"),
        navSchool = cms.ESInputTag("","SimpleNavigationSchool"),
        superClusters = cms.InputTag("hltEgammaSuperClustersToPixelMatchL1Seeded")
    )

    alpaka.toReplaceWith(process.hltEgammaElectronPixelSeedsL1Seeded, alpaka_converter)

    return process
