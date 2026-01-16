import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dtCertificationSummary = DQMEDHarvester('DTCertificationSummary',
                                        inputMEs = cms.untracked.VInputTag(("dtChamberEfficiencyClient"), ("dtResolutionAnalysisTest"), ("segmentTest")))


