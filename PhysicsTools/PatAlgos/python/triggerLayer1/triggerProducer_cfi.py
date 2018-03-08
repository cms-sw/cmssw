import FWCore.ParameterSet.Config as cms

patTrigger = cms.EDProducer( "PATTriggerProducer"
, onlyStandAlone = cms.bool( False )
,l1GtRecordInputTag = cms.InputTag("gtDigis")
,l1GtReadoutRecordInputTag = cms.InputTag("gtDigis")
,l1GtTriggerMenuLiteInputTag = cms.InputTag("gtDigis")
,l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis")
,l1tExtBlkInputTag = cms.InputTag("gtStage2Digis")
,ReadPrescalesFromFile = cms.bool(False)

# ## L1
# , addL1Algos                     = cms.bool( False )                                 # default; possibly superseded by 'onlyStandAlone' = True
# , l1GlobalTriggerObjectMaps      = cms.InputTag( "l1L1GtObjectMap" )                 # default; change only, if you know exactly, what you are doing!
# , l1ExtraMu                      = cms.InputTag( "l1extraParticles", ""            ) # default; change only, if you know exactly, what you are doing!
# , l1ExtraNoIsoEG                 = cms.InputTag( "l1extraParticles", "NonIsolated" ) # default; change only, if you know exactly, what you are doing!
# , l1ExtraIsoEG                   = cms.InputTag( "l1extraParticles", "Isolated"    ) # default; change only, if you know exactly, what you are doing!
# , l1ExtraCenJet                  = cms.InputTag( "l1extraParticles", "Central"     ) # default; change only, if you know exactly, what you are doing!
# , l1ExtraForJet                  = cms.InputTag( "l1extraParticles", "Forward"     ) # default; change only, if you know exactly, what you are doing!
# , l1ExtraTauJet                  = cms.InputTag( "l1extraParticles", "Tau"         ) # default; change only, if you know exactly, what you are doing!
# , l1ExtraETM                     = cms.InputTag( "l1extraParticles", "MET"         ) # default; change only, if you know exactly, what you are doing!
# , l1ExtraHTM                     = cms.InputTag( "l1extraParticles", "MHT"         ) # default; change only, if you know exactly, what you are doing!
# , mainBxOnly                     = cms.bool( True )                                  # default
# , saveL1Refs                     = cms.bool( False )                                 # default; setting this to True requires to keep '*_l1extraParticles_*_[processName]' and '*_gctDigis_*_[processName]' in the event
## HLT (L3)
, processName    = cms.string( "HLT" )                    # default; change only, if you know exactly, what you are doing!
# , triggerResults = cms.InputTag( "TriggerResults" )       # default; change only, if you know exactly, what you are doing!
# , triggerEvent   = cms.InputTag( "hltTriggerSummaryAOD" ) # default; change only, if you know exactly, what you are doing!
# , hltPrescaleLabel = cms.string( "0" )
# , hltPrescaleTable = cms.string( "hltPrescaleRecorder" )  # only the label!
# , addPathModuleLabels = cms.bool( False )                 # setting this "True" stores the names of all modules as strings (~10kB/ev.); possibly superseded by 'onlyStandAlone' = True
# , exludeCollections = cms.vstring()
, packTriggerLabels = cms.bool(False)
)

