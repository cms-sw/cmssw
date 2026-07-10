from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
es_prefer_l1GtTriggerMaskAlgoTrig = cms.ESPrefer("L1GtTriggerMaskAlgoTrigTrivialProducer","l1GtTriggerMaskAlgoTrig")
es_prefer_l1GtTriggerMaskTechTrig = cms.ESPrefer("L1GtTriggerMaskTechTrigTrivialProducer","l1GtTriggerMaskTechTrig")

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

hltL1SingleMuOpen = hltHighLevel.clone(
  HLTPaths = ['HLT_L1SingleMuOpen*']
)

hltDtCalibTest = hltHighLevel.clone(
  HLTPaths = ['HLT_Mu50_v*', 'HLT_IsoMu*', 'HLT_Mu13_Mu8_v*', 'HLT_Mu17_Mu8_v*']
)
hltDTCalibration = hltHighLevel.clone(
  HLTPaths = ['HLT_DTCalibration_v*']
)
ALCARECODtCalibHIHLTFilter = hltHighLevel.clone(
  HLTPaths = ['HLT_HIL2SingleMu*']
  #HLTPaths = ['HLT_OxyL1SingleMu*']
)

from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1tech = hltLevel1GTSeed.clone(
  L1TechTriggerSeeding = True
)
l1Algo = hltLevel1GTSeed.clone(
  L1TechTriggerSeeding = False
)
bptx = l1tech.clone(
  L1SeedsLogicalExpression = '0'
)
bscAnd = l1tech.clone(
  L1SeedsLogicalExpression = '40 OR 41'
)
beamHaloVeto = l1tech.clone(
  L1SeedsLogicalExpression = 'NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))'
)
l1SingleMuOpen = l1Algo.clone(
  L1SeedsLogicalExpression = 'L1_SingleMuOpen'
)

l1Coll = cms.Sequence(bptx)
l1CollBscAnd = cms.Sequence(bptx + bscAnd + beamHaloVeto)

primaryVertexFilter = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"),
    filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)

primaryVertexFilterHI = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"),
    filter = cms.bool(True),
)

scrapingEvtFilter = cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False),
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
)

hltDTActivityFilter = cms.EDFilter( "HLTDTActivityFilter",
    inputDCC         = cms.InputTag( "dttfDigis" ),
    inputDDU         = cms.InputTag( "muonDTDigis" ),
    inputDigis       = cms.InputTag( "muonDTDigis" ),
    processDCC       = cms.bool( False ),
    processDDU       = cms.bool( False ),
    processDigis     = cms.bool( True ),
    processingMode   = cms.int32( 0 ),   # 0=(DCC | DDU) | Digis/ 
                                         # 1=(DCC & DDU) | Digis/
                                         # 2=(DCC | DDU) & Digis/
                                         # 3=(DCC & DDU) & Digis/   
    minChamberLayers = cms.int32( 6 ),
    maxStation       = cms.int32( 3 ),
    minQual          = cms.int32( 2 ),   # 0-1=L 2-3=H 4=LL 5=HL 6=HH/
    minDDUBX         = cms.int32( 9 ),
    maxDDUBX         = cms.int32( 14 ),
    minActiveChambs  = cms.int32( 1 ),
    activeSectors    = cms.vint32(1,2,3,4,5,6,7,8,9,10,11,12)
)

#from CalibMuon.DTCalibration.DTCalibMuonSelection_cfi import *

goodMuonsPt15 = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('(isGlobalMuon = 1 | isTrackerMuon = 1) & abs(eta) < 1.2 & pt > 15.0'),
    filter = cms.bool(True) 
)
muonSelectionPt15 = cms.Sequence(goodMuonsPt15)

goodMuonsPt5 = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('(isGlobalMuon = 1 | isTrackerMuon = 1) & abs(eta) < 1.2 & pt > 5.0'),
    filter = cms.bool(True)
)
muonSelectionPt5 = cms.Sequence(goodMuonsPt5)

goodCosmicTracksPt5 = cms.EDFilter("TrackSelector",
    src = cms.InputTag("cosmicMuons"),
    cut = cms.string('pt > 5.0'),
    filter = cms.bool(True)
)


offlineSelection = cms.Sequence(scrapingEvtFilter + primaryVertexFilter + muonSelectionPt15)
offlineSelectionALCARECO = cms.Sequence(muonSelectionPt15)
offlineSelectionALCARECODtCalibTest = cms.Sequence(hltDtCalibTest + muonSelectionPt15)
offlineSelectionCosmics = cms.Sequence(hltL1SingleMuOpen)
offlineSelectionHI = cms.Sequence(ALCARECODtCalibHIHLTFilter + primaryVertexFilter + muonSelectionPt5)
offlineSelectionHIALCARECO = cms.Sequence(primaryVertexFilterHI + muonSelectionPt5)
offlineSelectionHIRAW = cms.Sequence(ALCARECODtCalibHIHLTFilter)
offlineSelectionTestEnables = cms.Sequence(hltDTCalibration)

dtCalibOfflineSelection = cms.Sequence(offlineSelection)
dtCalibOfflineSelectionALCARECO = cms.Sequence(offlineSelectionALCARECO)
dtCalibOfflineSelectionALCARECODtCalibTest = cms.Sequence(offlineSelectionALCARECODtCalibTest)
dtCalibOfflineSelectionCosmics = cms.Sequence(offlineSelectionCosmics)
dtCalibOfflineSelectionHI = cms.Sequence(offlineSelectionHI)
dtCalibOfflineSelectionHIALCARECO = cms.Sequence(offlineSelectionHIALCARECO)
dtCalibOfflineSelectionHIRAW = cms.Sequence(offlineSelectionHIRAW)
dtCalibOfflineSelectionTestEnables = cms.Sequence(offlineSelectionTestEnables)
