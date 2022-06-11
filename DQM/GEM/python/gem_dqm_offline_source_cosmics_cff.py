import FWCore.ParameterSet.Config as cms

from DQM.GEM.GEMDigiSource_cfi import *
from DQM.GEM.GEMRecHitSource_cfi import *
from DQM.GEM.gemEfficiencyAnalyzerCosmics_cff import *

GEMDigiSource.modeRelVal = True
GEMRecHitSource.modeRelVal = True

gemSourcesCosmics = cms.Sequence(
    GEMDigiSource *
    GEMRecHitSource *
    gemEfficiencyAnalyzerCosmicsGlb *
    gemEfficiencyAnalyzerCosmicsSta
)
