#
import FWCore.ParameterSet.Config as cms

l1tGlobalSummary = cms.EDAnalyzer("L1TGlobalSummary",
		                  AlgInputTag = cms.InputTag("gtStage2Digis"),
                                  ExtInputTag = cms.InputTag("gtStage2Digis"),
		                  ## ExtInputTag = cms.InputTag("gtExtFakeProd"),
		                  MinBx          = cms.int32(-2),
		                  MaxBx          = cms.int32(2),
		                  DumpRecord   = cms.bool(False), # print raw uGT record
                                  DumpTrigResults= cms.bool(False),
                                  DumpTrigSummary= cms.bool(True),
                                  ReadPrescalesFromFile= cms.bool(False)
)
