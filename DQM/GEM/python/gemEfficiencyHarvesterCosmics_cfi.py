import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.GEM.gemEfficiencyHarvesterDefault_cfi import gemEfficiencyHarvesterDefault as _gemEfficiencyHarvesterDefault
from DQM.GEM.gemEfficiencyAnalyzerCosmics_cfi import gemEfficiencyAnalyzerCosmics as _gemEfficiencyAnalyzerCosmics
from DQM.GEM.gemEfficiencyAnalyzerCosmics_cfi import gemEfficiencyAnalyzerCosmicsOneLeg as _gemEfficiencyAnalyzerCosmicsOneLeg

gemEfficiencyHarvesterCosmics = _gemEfficiencyHarvesterDefault.clone(
    folder = _gemEfficiencyAnalyzerCosmics.folder.value()
)

gemEfficiencyHarvesterCosmicsOneLeg = _gemEfficiencyHarvesterDefault.clone(
    folder = _gemEfficiencyAnalyzerCosmicsOneLeg.folder.value()
)
