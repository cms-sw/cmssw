import FWCore.ParameterSet.Config as cms

tbunpack = cms.EDFilter("HcalTBObjectUnpacker",
                                IncludeUnmatchedHits = cms.untracked.bool(False),
                                ConfigurationFile=cms.untracked.string("RecoTBCalo/HcalTBObjectUnpacker/data/configQADCTDC.txt"),
                                HcalTDCFED = cms.untracked.int32(8),
                                HcalQADCFED = cms.untracked.int32(8),
                                HcalSlowDataFED = cms.untracked.int32(3),
                                HcalTriggerFED = cms.untracked.int32(1)
                        )


