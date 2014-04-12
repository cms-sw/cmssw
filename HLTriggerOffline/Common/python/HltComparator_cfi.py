import FWCore.ParameterSet.Config as cms

hltComparator = cms.EDAnalyzer('HltComparator',
    OnlineResults = cms.InputTag( 'TriggerResults','','HLT' ),
    OfflineResults = cms.InputTag( 'TriggerResults','','HltRerun' ),
    verbose = cms.untracked.bool(False),
    skipPaths = cms.untracked.vstring('AlCa_RPCMuonNormalisation', 
                                      'HLT_Random','HLT_L1MuOpenPrescaled', 
                                      'HLT_PhysicsNoMuonPrescaled',
                                      'HLT_ZeroBiasPrescaled', 
                                      'HLT_PixelFEDSize'),
    usePaths = cms.untracked.vstring() 
 ) 
