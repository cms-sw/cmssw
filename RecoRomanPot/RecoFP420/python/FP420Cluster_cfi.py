import FWCore.ParameterSet.Config as cms

FP420Cluster = cms.EDProducer("ClusterizerFP420", #
    ROUList = cms.vstring('FP420Digi'),           #
    VerbosityLevel = cms.untracked.int32(0),      #
    NumberFP420Detectors = cms.int32(3),                   ## =3 means 2 Trackers: +FP420 and -FP420; =0 -> no FP420 at all 
    NumberFP420Stations = cms.int32(3),                    ## means 2 Stations w/ arm 8m
    NumberFP420SPlanes = cms.int32(6),                     ## means 5 SuperPlanes
    ClusterModeFP420 = cms.string('ClusterProducerFP420'), #
    ElectronFP420PerAdc = cms.double(300.0),               #
    ChannelFP420Threshold = cms.double(8.0),               #  5.0
    ClusterFP420Threshold = cms.double(9.0),               #  6.0
    SeedFP420Threshold = cms.double(8.5),                  #  5.5
    MaxVoidsFP420InCluster = cms.int32(1),                 #  1
    NumberHPS240Detectors = cms.int32(3),                      ## =3 means 2 Trackers: +HPS240 and -HPS240; =0 -> no HPS240 at all 
    NumberHPS240Stations = cms.int32(3),                       ## means 2 Stations w/ arm 8m
    NumberHPS240SPlanes = cms.int32(6),                        ## means 5 SuperPlanes
    ClusterModeHPS240 = cms.string('ClusterProducerHPS240'),   #
    ElectronHPS240PerAdc = cms.double(300.0),                  #
    ChannelHPS240Threshold = cms.double(8.0),                  #  5.0
    ClusterHPS240Threshold = cms.double(9.0),                  #  6.0
    SeedHPS240Threshold = cms.double(8.5),                     #  5.5
    MaxVoidsHPS240InCluster = cms.int32(1)                     #  1
)

