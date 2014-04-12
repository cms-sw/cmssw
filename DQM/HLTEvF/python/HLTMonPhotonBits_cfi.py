import FWCore.ParameterSet.Config as cms

# Bit Plotting
hltMonPhotonBits = cms.EDAnalyzer("HLTMonBitSummary",
     directory = cms.untracked.string("HLT/HLTMonPhoton/Summary"),
     #label = cms.string('myLabel'),
     #out = cms.untracked.string('dqm.root'),
     HLTPaths = cms.vstring('HLT_L1SingleEG5', 'HLT_Photon10_L1R', 'HLT_L1SingleEG8' ,
                            'HLT_L1DoubleEG5' , 'HLT_Photon15_L1R' ,
                            'HLT_Photon15_TrackIso_L1R' , 'HLT_Photon15_LooseEcalIso_L1R' ,
                            'HLT_Photon20_L1R' , 'HLT_Photon30_L1R_8E29' ,
                            'HLT_DoublePhoton10_L1R'
                            ),
                                  #FIXME - change to photon paths
     filterTypes = cms.untracked.vstring( "HLTLevel1GTSeed",
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

