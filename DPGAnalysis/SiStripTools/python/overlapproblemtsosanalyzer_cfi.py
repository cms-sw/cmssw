import FWCore.ParameterSet.Config as cms

overlapproblemtsosanalyzer = cms.EDAnalyzer("OverlapProblemTSOSAnalyzer",
                                            trajTrackAssoCollection = cms.InputTag("refittedTracks"),
                                            onlyValidRecHit = cms.bool(True),
                                            tsosHMConf = cms.PSet(
    wantedSubDets = cms.VPSet(cms.PSet(detLabel=cms.string("TECR1"),title=cms.string("TEC R1"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000020")),
                              cms.PSet(detLabel=cms.string("TECR2"),title=cms.string("TEC R2"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000040")),
                              cms.PSet(detLabel=cms.string("TECR3"),title=cms.string("TEC R3"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000060")),
                              cms.PSet(detLabel=cms.string("TECR4"),title=cms.string("TEC R4"),selection=cms.untracked.vstring("0x1e0000e0-0x1c000080")),
                              cms.PSet(detLabel=cms.string("TECR5"),title=cms.string("TEC R5"),selection=cms.untracked.vstring("0x1e0000e0-0x1c0000a0")),
                              cms.PSet(detLabel=cms.string("TECR6"),title=cms.string("TEC R6"),selection=cms.untracked.vstring("0x1e0000e0-0x1c0000c0")),
                              cms.PSet(detLabel=cms.string("TECR7"),title=cms.string("TEC R7"),selection=cms.untracked.vstring("0x1e0000e0-0x1c0000e0")),
                              cms.PSet(detLabel=cms.string("FPIXpP1"),title=cms.string("FPIX+ panel 1"),selection=cms.untracked.vstring("0x1f800300-0x15000100")),
                              cms.PSet(detLabel=cms.string("FPIXpP2"),title=cms.string("FPIX+ panel 2"),selection=cms.untracked.vstring("0x1f800300-0x15000200")),
                              cms.PSet(detLabel=cms.string("FPIXmP1"),title=cms.string("FPIX- panel 1"),selection=cms.untracked.vstring("0x1f800300-0x14800100")),
                              cms.PSet(detLabel=cms.string("FPIXmP2"),title=cms.string("FPIX- panel 2"),selection=cms.untracked.vstring("0x1f800300-0x14800200"))
                              )
    )
                                            )

