import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.GEM.gemEfficiencyHarvesterDefault_cfi import gemEfficiencyHarvesterDefault as _gemEfficiencyHarvesterDefault
from DQM.GEM.gemEfficiencyAnalyzer_cfi import gemEfficiencyAnalyzerTightGlb as _gemEfficiencyAnalyzerTightGlb
from DQM.GEM.gemEfficiencyAnalyzer_cfi import gemEfficiencyAnalyzerSta as _gemEfficiencyAnalyzerSta

gemEfficiencyHarvesterTightGlb = _gemEfficiencyHarvesterDefault.clone(
    folder = _gemEfficiencyAnalyzerTightGlb.folder.value()
)

gemEfficiencyHarvesterSta = _gemEfficiencyHarvesterDefault.clone(
    folder = _gemEfficiencyAnalyzerSta.folder.value()
)
