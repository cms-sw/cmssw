import FWCore.ParameterSet.Config as cms

from RecoBTag.PixelCluster.pixelClusterTagInfos_cfi import *

pixelClusterTask = cms.Task(
    pixelClusterTagInfos,
)

