import FWCore.ParameterSet.Config as cms

HLTTauValPostAnalysis = cms.EDFilter("HLTTauPostProcessor",
                            L1Folder = cms.vstring('HLT/HLTTAU/L1'),
                            L2Folder = cms.vstring('HLT/HLTTAU/L2'),

                            L25Folder = cms.vstring('HLT/HLTTAU/L25'),

                            L3Folder = cms.vstring('HLT/HLTTAU/L3'),
                                                  
                            
                            
                            HLTPathValidationFolder = cms.vstring('HLT/HLTTAU/DoubleTau',
                                                   'HLT/HLTTAU/SingleTau',
                                                   'HLT/HLTTAU/ElectronTau',
                                                   'HLT/HLTTAU/MuonTau'
                                                   ),
    
                            HLTPathDQMFolder = cms.vstring('')
                            )


HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis)
