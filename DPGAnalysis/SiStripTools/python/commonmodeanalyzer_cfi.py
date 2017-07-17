import FWCore.ParameterSet.Config as cms

commonmodeanalyzer = cms.EDAnalyzer('CommonModeAnalyzer',
                                    digiCollection = cms.InputTag("siStripDigis","CommonMode"),
                                    badModuleDigiCollection = cms.InputTag("siStripDigis"),
                                    historyProduct = cms.InputTag("consecutiveHEs"),
                                    apvPhaseCollection = cms.InputTag("APVPhases"),
                                    phasePartition = cms.untracked.string("All"),
                                    ignoreBadFEDMod = cms.bool(True),
                                    ignoreNotConnected = cms.bool(True),
                                    selections = cms.VPSet(
        cms.PSet(label=cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),
        cms.PSet(label=cms.string("TEC"),selection=cms.untracked.vstring("0x1e000000-0x1c000000")),
        cms.PSet(label=cms.string("TOB"),selection=cms.untracked.vstring("0x1e000000-0x1a000000")),
        cms.PSet(label=cms.string("TID"),selection=cms.untracked.vstring("0x1e000000-0x18000000"))
        )
                                    )	
