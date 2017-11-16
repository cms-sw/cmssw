import FWCore.ParameterSet.Config as cms

DTDataIntegrityTask = cms.EDAnalyzer("DTDataIntegrityTask",
                                     getSCInfo = cms.untracked.bool(True),
                                     checkUros  = cms.untracked.bool(True),
				     fedIDMIN  = cms.untracked.int32(1368),
				     fedIDMAX = cms.untracked.int32(1370),
                                     fedIntegrityFolder = cms.untracked.string("DT/FEDIntegrity"),
                                     processingMode     = cms.untracked.string("Online"),
                                     dtDDULabel         = cms.InputTag("dtDataIntegrityUnpacker"),
                                     dtROS25Label       = cms.InputTag("dtDataIntegrityUnpacker"),
				     dtFEDlabel         =  cms.InputTag("dtDataIntegrityUnpacker")
)


