import FWCore.ParameterSet.Config as cms

# Bit Plotting
hltMonOniaBits = cms.EDAnalyzer("HLTMonBitSummary",
     directory = cms.untracked.string('HLT/HLTMonMuon/Onia/Summary/'),
     histLabel = cms.untracked.string('Onia'),
     HLTPaths = cms.vstring('HLT_L1MuOpen',
                            'HLT_Mu0_Track0_Jpsi',
                            'HLT_Mu3_Track0_Jpsi',
                            'HLT_Mu5_Track0_Jpsi'
                            ),
                              
     filterTypes = cms.untracked.vstring("HLTMuonL3PreFilter",
                                         "HLTMuonTrackMassFilter"
                                        ),
                              

    denominatorWild = cms.untracked.string('HLT_L1MuOpen'),
    denominator = cms.untracked.string('HLT_L1MuOpen'),
    
    TriggerResultsTag = cms.InputTag('TriggerResults','','HLT'),

)
