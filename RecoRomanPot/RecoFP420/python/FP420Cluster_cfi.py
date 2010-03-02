import FWCore.ParameterSet.Config as cms

FP420Cluster = cms.EDProducer("ClusterizerFP420",
    ROUList = cms.vstring('FP420Digi'),
    VerbosityLevel = cms.untracked.int32(0),
    NumberFP420Stations = cms.int32(3),
    NumberFP420Detectors = cms.int32(3),
    NumberFP420SPlanes = cms.int32(6),
    ClusterModeFP420 = cms.string('ClusterProducerFP420'),
    ElectronFP420PerAdc = cms.double(300.0),
    ChannelFP420Threshold = cms.double(8.0),
    ClusterFP420Threshold = cms.double(9.0),
    SeedFP420Threshold = cms.double(8.5),
    MaxVoidsFP420InCluster = cms.int32(1)
)



