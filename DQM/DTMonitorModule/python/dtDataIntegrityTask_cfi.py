import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
DTDataIntegrityTask = DQMEDAnalyzer('DTDataIntegrityTask',
                                     getSCInfo = cms.untracked.bool(True),
                                     checkUros  = cms.untracked.bool(False),
                                     fedIntegrityFolder = cms.untracked.string("DT/FEDIntegrity"),
                                     processingMode     = cms.untracked.string("Online"),
                                     dtDDULabel         = cms.InputTag("dtDataIntegrityUnpacker"),
                                     dtROS25Label       = cms.InputTag("dtDataIntegrityUnpacker"),
				     dtFEDlabel         =  cms.InputTag("dtDataIntegrityUnpacker")
)

from Configuration.Eras.Modifier_run2_DT_2018_cff import run2_DT_2018
run2_DT_2018.toModify(DTDataIntegrityTask,checkUros=cms.untracked.bool(True))

