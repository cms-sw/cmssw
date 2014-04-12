import FWCore.ParameterSet.Config as cms

valueMapChecker = cms.EDAnalyzer("GEDValueMapAnalyzer",
                                 PFCandidates = cms.InputTag("particleFlowEGamma"),
                                 ElectronValueMap = cms.InputTag("gedGsfElectrons"))
