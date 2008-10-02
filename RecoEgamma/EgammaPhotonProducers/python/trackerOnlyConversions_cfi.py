import FWCore.ParameterSet.Config as cms

trackerOnlyConversions = cms.EDProducer('TrackerOnlyConversionProducer',
    src = cms.VInputTag(cms.InputTag("generalTracks")),
    convertedPhotonCollection = cms.string(''), ## or empty

    bcEndcapCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),

    HalfwayEta = cms.double(.1),
    dEtaTrackBC = cms.double(.06),
    dPhiTrackBC = cms.double(.6),
    EnergyBC = cms.double(1.5),
    MaxChi2Left = cms.double(30.),
    MaxChi2Right = cms.double(30.),
    MaxHitsLeft = cms.int32(5),
    MaxHitsRight = cms.int32(2),
    DeltaCotTheta = cms.double(.02),
    DeltaPhi = cms.double(.2),
    AllowSingleLeg = cms.bool(True)
)
