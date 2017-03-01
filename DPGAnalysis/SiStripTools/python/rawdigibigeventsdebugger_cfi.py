import FWCore.ParameterSet.Config as cms

rawdigibigeventsdebugger = cms.EDAnalyzer('RawDigiBigEventsDebugger',
                                          collection = cms.InputTag("siStripDigis","VirginRaw"),
                                          singleEvents = cms.bool(False),
                                          foldedStrips = cms.untracked.bool(False),
                                          want1dHisto = cms.untracked.bool(True),
                                          wantProfile = cms.untracked.bool(True),
                                          want2dHisto = cms.untracked.bool(True),
                                          selections = cms.VPSet(
    cms.PSet(label=cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),
    cms.PSet(label=cms.string("TEC"),selection=cms.untracked.vstring("0x1e000000-0x1c000000")),
    cms.PSet(label=cms.string("TOB"),selection=cms.untracked.vstring("0x1e000000-0x1a000000")),
    cms.PSet(label=cms.string("TID"),selection=cms.untracked.vstring("0x1e000000-0x18000000"))
    )

)
