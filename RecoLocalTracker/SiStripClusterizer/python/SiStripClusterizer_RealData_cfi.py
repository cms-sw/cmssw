# The following comments couldn't be translated into the new config version:

#                { string DigiProducer = "stripdigi"
#                  string DigiLabel    = "\0"
#                },

import FWCore.ParameterSet.Config as cms

siStripClusters = cms.EDFilter("SiStripClusterizer",
    MaxHolesInCluster = cms.int32(0),
    ChannelThreshold = cms.double(2.0),
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('ZeroSuppressed'),
        DigiProducer = cms.string('SiStripDigis')
    ), cms.PSet(
        DigiLabel = cms.string('VirginRaw'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ), cms.PSet(
        DigiLabel = cms.string('ProcessedRaw'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ), cms.PSet(
        DigiLabel = cms.string('ScopeMode'),
        DigiProducer = cms.string('siStripZeroSuppression')
    )),
    ClusterMode = cms.string('ThreeThresholdClusterizer'),
    SeedThreshold = cms.double(3.0),
    SiStripQualityLabel = cms.string(''),
    ClusterThreshold = cms.double(5.0)
)


