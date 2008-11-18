import FWCore.ParameterSet.Config as cms

HLTTauValPostAnalysis = cms.EDFilter("HLTTauPostProcessor",
                            L1Folder = cms.vstring('HLT/HLTTAU/L1'),
                            L2Folder = cms.vstring('HLT/HLTTAU/DoubleTau/L2',
                                                   'HLT/HLTTAU/SingleTau/L2',
                                                   'HLT/HLTTAU/SingleTauMET/L2',
                                                   'HLT/HLTTAU/ElectronTau/L2',
                                                   'HLT/HLTTAU/MuonTau/L2'
                                                   ),

                            L25Folder = cms.vstring('HLT/HLTTAU/DoubleTau/L25',
                                                   'HLT/HLTTAU/SingleTau/L25',
                                                   'HLT/HLTTAU/SingleTauMET/L25',
                                                   'HLT/HLTTAU/ElectronTau/L25',
                                                   'HLT/HLTTAU/MuonTau/L25'
                                                   ),

                            L3Folder = cms.vstring('HLT/HLTTAU/SingleTau/L3',
                                                  'HLT/HLTTAU/SingleTauMET/L3'
                                                  ),
                                                  
                            
                            
                            HLTPathValidationFolder = cms.vstring('HLT/HLTTAU/DoubleTau/Path',
                                                   'HLT/HLTTAU/SingleTau/Path',
                                                   'HLT/HLTTAU/SingleTauMET/Path',
                                                   'HLT/HLTTAU/ElectronTau/Path',
                                                   'HLT/HLTTAU/MuonTau/Path'
                                                   ),
    
                            HLTPathDQMFolder = cms.vstring('')
                            )


HLTTauPostVal = cms.Sequence(HLTTauValPostAnalysis)
