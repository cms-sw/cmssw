import FWCore.ParameterSet.Config as cms

from DQM.GEM.GEMDigiSource_cfi import *
from DQM.GEM.GEMRecHitSource_cfi import *
from DQM.GEM.GEMDAQStatusSource_cfi import *
from DQM.GEM.gemEfficiencyAnalyzerCosmics_cff import *

GEMDigiSource.runType      = "offline"
GEMRecHitSource.runType    = "offline"
GEMDAQStatusSource.runType = "offline"

gemSourcesCosmics = cms.Sequence(
    GEMDigiSource *
    GEMRecHitSource *
    GEMDAQStatusSource *
    gemEfficiencyAnalyzerCosmicsGlb *
    gemEfficiencyAnalyzerCosmicsSta
)
