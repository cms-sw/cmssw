import FWCore.ParameterSet.Config as cms

from DQM.GEM.GEMDigiSource_cfi import *
from DQM.GEM.GEMRecHitSource_cfi import *

from DQMOffline.Muon.gemEfficiencyAnalyzer_cfi import *

GEMDigiSource.modeRelVal = True
GEMRecHitSource.modeRelVal = True

gemSources = cms.Sequence(
    GEMDigiSource *
    GEMRecHitSource *
    gemEfficiencyAnalyzerTightGlbSeq *
    gemEfficiencyAnalyzerStaSeq
)
