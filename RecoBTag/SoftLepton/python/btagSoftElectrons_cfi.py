import FWCore.ParameterSet.Config as cms

# identify electrons from the ECAL energy deposits associate to the tracks
from TrackingTools.TrackAssociator.default_cfi import *
btagSoftElectrons = cms.EDProducer("SoftElectronProducer",
    TrackAssociatorParameterBlock,
    TrackTag = cms.InputTag("generalTracks"),
    BasicClusterTag = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
#    BasicClusterShapeTag = cms.InputTag("hybridSuperClusters"),
    BarrelRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
    EndcapRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),

    HBHERecHitTag = cms.InputTag("hbhereco"),
    DiscriminatorCut = cms.double(0.9),
    HOverEConeSize = cms.double(0.3)
)

btagSoftElectrons.TrackAssociatorParameters.useEcal = True
btagSoftElectrons.TrackAssociatorParameters.useHcal = False
btagSoftElectrons.TrackAssociatorParameters.useHO   = False
btagSoftElectrons.TrackAssociatorParameters.useMuon = False
