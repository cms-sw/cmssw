import FWCore.ParameterSet.Config as cms

# Bit Plotting
hltMonEleBits = cms.EDAnalyzer("HLTMonBitSummary",
     directory = cms.untracked.string("HLT/HLTMonElectron/"),
     histLabel = cms.untracked.string("Electron"),
     #label = cms.string('myLabel'),
     #out = cms.untracked.string('dqm.root'),
     HLTPaths = cms.vstring('HLT_L1SingleEG',
                            'HLT_Ele',
                            'HLT_L1DoubleEG',
                            'HLT_DoubleEle'
                            ),
     filterTypes = cms.vstring( "HLTLevel1GTSeed",
                                "HLTPrescaler",
                                "HLTEgammaL1MatchFilterRegional",
                                "HLTEgammaEtFilter",
                                "HLTEgammaGenericFilter",
                                "HLTElectronPixelMatchFilter",
                                "HLTElectronOneOEMinusOneOPFilterRegional"
                               ),
    denominatorWild = cms.untracked.string('HLT_L1SingleEG'),                           
    denominator = cms.untracked.string('HLT_L1SingleEG5'),
    
    TriggerResultsTag = cms.InputTag('TriggerResults','','HLT'),

)

