import FWCore.ParameterSet.Config as cms

# raw-to-clusters facility
from EventFilter.SiStripRawToDigi.SiStripRawToClusters_cfi import *
import copy
from EventFilter.SiStripRawToDigi.SiStripRawToClustersRoI_cfi import *
# raw-to-clusters regions of interest module
siStripClusters = copy.deepcopy(SiStripRoI)
SiStripRawToClusters = cms.Sequence(SiStripRawToClustersFacility*siStripClusters)
