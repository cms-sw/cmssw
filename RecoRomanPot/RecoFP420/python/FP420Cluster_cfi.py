import FWCore.ParameterSet.Config as cms

FP420Cluster = cms.EDFilter("ClusterizerFP420",
    MaxVoidsFP420InCluster = cms.int32(1),
    NumberFP420Stations = cms.int32(3),
    #--------------------------------
    #--------------------------------
    ROUList = cms.vstring('FP420Digi'),
    #--------------------------------
    #-----------------------------ClusterizerFP420 
    #--------------------------------
    NumberFP420Detectors = cms.int32(3),
    ChannelFP420Threshold = cms.double(7.0),
    #--------------------------------
    #-----------------------------FP420ClusterMain
    #--------------------------------
    ElectronFP420PerAdc = cms.double(300.0),
    ClusterFP420Threshold = cms.double(8.0),
    ClusterModeFP420 = cms.string('ClusterProducerFP420'),
    SeedFP420Threshold = cms.double(7.5),
    #--------------------------------
    #--------------------------------
    VerbosityLevel = cms.untracked.int32(0),
    NumberFP420SPlanes = cms.int32(6)
)


