import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.Muon.gemEfficiencyHarvesterDefault_cfi import gemEfficiencyHarvesterDefault as _gemEfficiencyHarvesterDefault
from DQMOffline.Muon.gemEfficiencyAnalyzerCosmics_cfi import gemEfficiencyAnalyzerCosmics as _gemEfficiencyAnalyzerCosmics
from DQMOffline.Muon.gemEfficiencyAnalyzerCosmics_cfi import gemEfficiencyAnalyzerCosmicsOneLeg as _gemEfficiencyAnalyzerCosmicsOneLeg

gemEfficiencyHarvesterCosmics = _gemEfficiencyHarvesterDefault.clone(
    folder = cms.untracked.string(_gemEfficiencyAnalyzerCosmics.folder.value()),
)

gemEfficiencyHarvesterCosmicsOneLeg = _gemEfficiencyHarvesterDefault.clone(
    folder = cms.untracked.string(_gemEfficiencyAnalyzerCosmicsOneLeg.folder.value()),
)
