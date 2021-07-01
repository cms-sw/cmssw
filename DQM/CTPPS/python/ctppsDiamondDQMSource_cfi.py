import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ctppsDiamondDQMSource = DQMEDAnalyzer('CTPPSDiamondDQMSource',
    tagStatus = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDigi = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagFEDInfo = cms.InputTag("ctppsDiamondRawToDigi", "TimingDiamond"),
    tagDiamondRecHits = cms.InputTag("ctppsDiamondRecHits"),
    tagDiamondLocalTracks = cms.InputTag("ctppsDiamondLocalTracks"),
    tagPixelLocalTracks = cms.InputTag("ctppsPixelLocalTracks"),
    
    excludeMultipleHits = cms.bool(True),

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
