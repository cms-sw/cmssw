import FWCore.ParameterSet.Config as cms

DTDataIntegrityTask = cms.EDAnalyzer("DTDataIntegrityTask",
                                     getSCInfo = cms.untracked.bool(True),
                                     checkUros  = cms.untracked.bool(False),
				     FEDIDmin  = cms.untracked.int32(770),
				     FEDIDmax = cms.untracked.int32(774),
                                     fedIntegrityFolder = cms.untracked.string("DT/FEDIntegrity"),
                                     processingMode     = cms.untracked.string("Online"),
                                     dtDDULabel         = cms.InputTag("dtDataIntegrityUnpacker"),
                                     dtROS25Label       = cms.InputTag("dtDataIntegrityUnpacker"),
				     dtFEDlabel         =  cms.InputTag("dtDataIntegrityUnpacker")
)

from Configuration.Eras.Modifier_run2_DT_2018_cff import run2_DT_2018
run2_DT_2018.toModify(DTDataIntegrityTask,FEDIDmin=cms.untracked.int32(1368))
run2_DT_2018.toModify(DTDataIntegrityTask,FEDIDmax=cms.untracked.int32(1370))
run2_DT_2018.toModify(DTDataIntegrityTask,checkUros=cms.untracked.bool(True))

