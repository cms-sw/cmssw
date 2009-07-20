import FWCore.ParameterSet.Config as cms

# Bit Plotting
hltMonEleBits = cms.EDAnalyzer("HLTMonBitSummary",
     directory = cms.untracked.string("HLT/HLTMonElectron/"),
     #label = cms.string('myLabel'),
     #out = cms.untracked.string('dqm.root'),
     HLTPaths = cms.vstring('HLT_L1SingleEG5','HLT_L1SingleEG8', 'HLT_Ele10_LW_L1R',
			    'HLT_Ele10_LW_EleId_L1R','HLT_Ele15_LW_L1R','HLT_Ele15_SiStrip_L1R',
                            'HLT_L1DoubleEG5','HLT_DoubleEle5_SW_L1R'
                            ),
     filterTypes = cms.vstring( "HLTLevel1GTSeed",
                                "HLTPrescaler",
                                "HLTEgammaL1MatchFilterRegional",
                                "HLTEgammaEtFilter",
                                "HLTEgammaGenericFilter",
                                "HLTElectronPixelMatchFilter",
                                "HLTElectronOneOEMinusOneOPFilterRegional"
                               ),
    denominator = cms.untracked.string('HLT_L1SingleEG5'),
    
    TriggerResultsTag = cms.InputTag('TriggerResults','','HLT'),

)

