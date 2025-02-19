from HLTrigger.special.hltPhysicsDeclared_cfi import *
hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'
hltbeamgas = cms.EDFilter("HLTHighLevel",
                          TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                          HLTPaths = cms.vstring('HLT_L1_BptxXOR_BscMinBiasOR'), # provide list of HLT paths (or patterns) you want
                          eventSetupPathsKey = cms.string(''),
                          andOr              = cms.bool(True),
                          throw              = cms.bool(False),
                          saveTags           = cms.bool(False)
                          )
pfgskim3noncrossseq = cms.Sequence(hltPhysicsDeclared*hltbeamgas)
