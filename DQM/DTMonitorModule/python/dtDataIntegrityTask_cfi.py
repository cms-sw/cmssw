import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
DTDataIntegrityTask = DQMEDAnalyzer('DTDataIntegrityTask',
                                     getSCInfo = cms.untracked.bool(True),
                                     fedIntegrityFolder = cms.untracked.string("DT/FEDIntegrity"),
                                     processingMode     = cms.untracked.string("Online"),
                                     dtDDULabel         = cms.InputTag("dtunpacker"),
                                     dtROS25Label       = cms.InputTag("dtunpacker")
)


