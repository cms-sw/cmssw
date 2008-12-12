import FWCore.ParameterSet.Config as cms

# raw-to-clusters facility
from EventFilter.SiStripRawToDigi.SiStripRawToClusters_cfi import *
# raw-to-clusters regions of interest module
from EventFilter.SiStripRawToDigi.SiStripRawToClustersRoI_cfi import *
# DetSetVector builder
siStripClusters = cms.EDProducer("SiStripClustersDSVBuilder",
    SiStripLazyGetter = cms.InputTag("SiStripRawToClustersFacility"),
    SiStripRefGetter = cms.InputTag("SiStripRoI")
)

SiStripRawToClusters = cms.Sequence(SiStripRawToClustersFacility*SiStripRoI*siStripClusters)

