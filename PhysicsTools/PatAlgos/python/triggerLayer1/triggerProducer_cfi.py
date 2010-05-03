import FWCore.ParameterSet.Config as cms

patTrigger = cms.EDProducer( "PATTriggerProducer"
                           , onlyStandAlone = cms.bool( False )
                           ## L1
                           , l1ExtraMu      = cms.InputTag( "l1extraParticles", ""           , "RECO" ) # default; change only, if you know exactly, what you are doing!
                           , l1ExtraNoIsoEG = cms.InputTag( "l1extraParticles", "NonIsolated", "RECO" ) # default; change only, if you know exactly, what you are doing!
                           , l1ExtraIsoEG   = cms.InputTag( "l1extraParticles", "Isolated"   , "RECO" ) # default; change only, if you know exactly, what you are doing!
                           , l1ExtraCenJet  = cms.InputTag( "l1extraParticles", "Central"    , "RECO" ) # default; change only, if you know exactly, what you are doing!
                           , l1ExtraForJet  = cms.InputTag( "l1extraParticles", "Forward"    , "RECO" ) # default; change only, if you know exactly, what you are doing!
                           , l1ExtraTauJet  = cms.InputTag( "l1extraParticles", "Tau"        , "RECO" ) # default; change only, if you know exactly, what you are doing!
                           , l1ExtraETM     = cms.InputTag( "l1extraParticles", "MET"        , "RECO" ) # default; change only, if you know exactly, what you are doing!
                           , l1ExtraHTM     = cms.InputTag( "l1extraParticles", "MHT"        , "RECO" ) # default; change only, if you know exactly, what you are doing!
                           ## HLT (L3)
                           , processName    = cms.string( 'HLT' )                    # default; change only, if you know exactly, what you are doing!
                           , triggerResults = cms.InputTag( "TriggerResults" )       # default; change only, if you know exactly, what you are doing!
                           , triggerEvent   = cms.InputTag( "hltTriggerSummaryAOD" ) # default; change only, if you know exactly, what you are doing!
                           , addPathModuleLabels = cms.bool( False )                 # setting this 'True' stores the names of all modules as strings (~10kB/ev.)
                           )

