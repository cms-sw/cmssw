import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ctppsDiamondDQMSource = DQMEDAnalyzer('CTPPSDiamondDQMSource',
    tagStatus = cms.untracked.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDigi = cms.untracked.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagFEDInfo = cms.untracked.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDiamondRecHits = cms.untracked.InputTag("ctppsDiamondRecHits"),
    tagDiamondLocalTracks = cms.untracked.InputTag("ctppsDiamondLocalTracks"),
    tagPixelLocalTracks = cms.untracked.InputTag("ctppsPixelLocalTracks"),

    excludeMultipleHits = cms.bool(True),
    extractDigiInfo = cms.bool(True),
    
    plotOnline = cms.untracked.bool(True),
    plotOffline= cms.untracked.bool(False),
    

    offsetsOOT = cms.VPSet( # cut on the OOT bin for physics hits
        # 2016, after TS2
        cms.PSet(
            validityRange = cms.EventRange("1:min - 292520:max"),
            centralOOT = cms.int32(1),
        ),
        # 2017
        cms.PSet(
            validityRange = cms.EventRange("292521:min - 301417:max"),
            centralOOT = cms.int32(3),
        ),
        # 2017, after channel delays corrections
        cms.PSet(
            validityRange = cms.EventRange("301418:min - 301517:max"),
            centralOOT = cms.int32(1),
        ),
        # 2017, after channel delays corrections
        cms.PSet(
            validityRange = cms.EventRange("301518:min - 9999999:max"),
            centralOOT = cms.int32(0),
        ),
    ),

    perLSsaving = cms.untracked.bool(False), #driven by DQMServices/Core/python/DQMStore_cfi.py

    verbosity = cms.untracked.uint32(10),
)

ctppsDiamondDQMOfflineSource = DQMEDAnalyzer('CTPPSDiamondDQMSource',
    tagStatus = cms.untracked.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDigi = cms.untracked.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagFEDInfo = cms.untracked.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDiamondRecHits = cms.untracked.InputTag("ctppsDiamondRecHits"),
    tagDiamondLocalTracks = cms.untracked.InputTag("ctppsDiamondLocalTracks"),
    tagPixelLocalTracks = cms.untracked.InputTag("ctppsPixelLocalTracks"),

    excludeMultipleHits = cms.bool(True),
    extractDigiInfo = cms.bool(True),

    plotOnline = cms.untracked.bool(False),
    plotOffline= cms.untracked.bool(True),
    
    offsetsOOT = cms.VPSet( # cut on the OOT bin for physics hits
        # 2016, after TS2
        cms.PSet(
            validityRange = cms.EventRange("1:min - 292520:max"),
            centralOOT = cms.int32(1),
        ),
        # 2017
        cms.PSet(
            validityRange = cms.EventRange("292521:min - 301417:max"),
            centralOOT = cms.int32(3),
        ),
        # 2017, after channel delays corrections
        cms.PSet(
            validityRange = cms.EventRange("301418:min - 301517:max"),
            centralOOT = cms.int32(1),
        ),
        # 2017, after channel delays corrections
        cms.PSet(
            validityRange = cms.EventRange("301518:min - 9999999:max"),
            centralOOT = cms.int32(0),
        ),
    ),

    perLSsaving = cms.untracked.bool(False), #driven by DQMServices/Core/python/DQMStore_cfi.py

    verbosity = cms.untracked.uint32(10),
)
