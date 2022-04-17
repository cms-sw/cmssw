import FWCore.ParameterSet.Config as cms

from DQM.GEM.gemEfficiencyHarvester_cfi import gemEfficiencyHarvester
from DQM.GEM.gemEfficiencyAnalyzerCosmics_cff import gemEfficiencyAnalyzerCosmicsTwoLeg as _gemEfficiencyAnalyzerCosmicsTwoLeg
from DQM.GEM.gemEfficiencyAnalyzerCosmics_cff import gemEfficiencyAnalyzerCosmicsOneLeg as _gemEfficiencyAnalyzerCosmicsOneLeg

gemEfficiencyHarvesterCosmicsTwoLeg = gemEfficiencyHarvester.clone(
    folder = _gemEfficiencyAnalyzerCosmicsTwoLeg.folder.value()
)

gemEfficiencyHarvesterCosmicsOneLeg = gemEfficiencyHarvester.clone(
    folder = _gemEfficiencyAnalyzerCosmicsOneLeg.folder.value()
)
