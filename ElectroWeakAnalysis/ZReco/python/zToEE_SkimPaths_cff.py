import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToEESequences_cff import *
from ElectroWeakAnalysis.ZReco.zToEEHLTFilter_cfi import *
from ElectroWeakAnalysis.ZReco.zToEEFilter_cfi import *
from ElectroWeakAnalysis.ZReco.zToEEOneTrackFilter_cfi import *
from ElectroWeakAnalysis.ZReco.zToEEOneSuperClusterFilter_cfi import *
zToEEPath = cms.Path(zToEEHLTFilter+electronRecoForZToEE+zToEE+zToEEFilter+electronMCTruthForZToEE)
zToEEOneTrackPath = cms.Path(zToEEHLTFilter+electronRecoForZToEE+zToEEOneTrack+zToEEOneTrackFilter+electronMCTruthForZToEEOneTrack)
zToEEOneSuperClusterPath = cms.Path(zToEEHLTFilter+electronRecoForZToEE+zToEEOneSuperCluster+zToEEOneSuperClusterFilter+electronMCTruthForZToEEOneSuperCluster)

