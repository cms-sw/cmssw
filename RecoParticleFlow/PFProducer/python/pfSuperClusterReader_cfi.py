import FWCore.ParameterSet.Config as cms

pfSuperClusterReader = cms.EDAnalyzer("PFSuperClusterReader",
                                      GSFTracks = cms.InputTag("electronGsfTracks"),
                                      SuperClusterRefMap = cms.InputTag("pfElectronTranslator:pf"),
                                      MVAMap = cms.InputTag("pfElectronTranslator:pf"),
                                      PFCandidate = cms.InputTag("particleFlowTmp:electrons")
                                      )
