

from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
es_prefer_l1GtTriggerMaskAlgoTrig = cms.ESPrefer("L1GtTriggerMaskAlgoTrigTrivialProducer","l1GtTriggerMaskAlgoTrig")
es_prefer_l1GtTriggerMaskTechTrig = cms.ESPrefer("L1GtTriggerMaskTechTrigTrivialProducer","l1GtTriggerMaskTechTrig")

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
hltL1SingleMuOpen = copy.deepcopy(hltHighLevel)
hltL1SingleMuOpen.HLTPaths = ['HLT_L1SingleMuOpen_AntiBPTX_v*']
hltDtCalibTest = copy.deepcopy(hltHighLevel)
hltDtCalibTest.HLTPaths = ['HLT_Mu40_v*', 'HLT_IsoMu*', 'HLT_Mu13_Mu8_v*', 'HLT_Mu17_Mu8_v*']
hltDTCalibration = copy.deepcopy(hltHighLevel)
hltDTCalibration.HLTPaths = ['HLT_DTCalibration_v*']

ALCARECODtCalibHIHLTFilter = copy.deepcopy(hltHighLevel)
ALCARECODtCalibHIHLTFilter.throw = False
ALCARECODtCalibHIHLTFilter.eventSetupPathsKey = 'MuAlcaDtCalibHI'

from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1tech = hltLevel1GTSeed.clone()
l1tech.L1TechTriggerSeeding = cms.bool(True)

l1Algo = hltLevel1GTSeed.clone()
l1Algo.L1TechTriggerSeeding = cms.bool(False)

bptx = l1tech.clone()
bptx.L1SeedsLogicalExpression = cms.string('0')

bscAnd = l1tech.clone()
bscAnd.L1SeedsLogicalExpression = cms.string('40 OR 41')

beamHaloVeto = l1tech.clone()
beamHaloVeto.L1SeedsLogicalExpression = cms.string('NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')

l1SingleMuOpen = l1Algo.clone()
l1SingleMuOpen.L1SeedsLogicalExpression = cms.string('L1_SingleMuOpen')

#l1Coll = cms.Sequence(bptx + beamHaloVeto)
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

offlineSelectionPt15 = cms.Sequence(scrapingEvtFilter + primaryVertexFilter + muonSelectionPt15)
offlineSelectionALCARECOPt15 = cms.Sequence(muonSelectionPt15)
offlineSelectionPt5 = cms.Sequence(scrapingEvtFilter + primaryVertexFilter + muonSelectionPt5)
offlineSelectionALCARECOPt5 = cms.Sequence(muonSelectionPt5)
offlineSelectionCosmicsPt5 = cms.Sequence(hltL1SingleMuOpen + goodCosmicTracksPt5)
offlineSelectionHIPt5 = cms.Sequence(ALCARECODtCalibHIHLTFilter + primaryVertexFilterHI + muonSelectionPt5)
offlineSelectionHIALCARECOPt5 = cms.Sequence(primaryVertexFilterHI + muonSelectionPt5)
offlineSelectionHIRAWPt5 = cms.Sequence(ALCARECODtCalibHIHLTFilter)

offlineSelection = cms.Sequence(scrapingEvtFilter + primaryVertexFilter + muonSelectionPt15)
offlineSelectionALCARECO = cms.Sequence(muonSelectionPt15)
offlineSelectionALCARECODtCalibTest = cms.Sequence(hltDtCalibTest + muonSelectionPt15)
offlineSelectionCosmics = cms.Sequence(hltL1SingleMuOpen)
offlineSelectionHI = cms.Sequence(offlineSelectionHIPt5)
offlineSelectionHIALCARECO = cms.Sequence(offlineSelectionHIALCARECOPt5)
offlineSelectionHIRAW = cms.Sequence(offlineSelectionHIRAWPt5)
offlineSelectionTestEnables = cms.Sequence(hltDTCalibration)

dtCalibOfflineSelectionPt15 = cms.Sequence(offlineSelectionPt15)
dtCalibOfflineSelectionALCARECOPt15 = cms.Sequence(offlineSelectionALCARECOPt15)
dtCalibOfflineSelectionPt5 = cms.Sequence(offlineSelectionPt5)
dtCalibOfflineSelectionALCARECOPt5 = cms.Sequence(offlineSelectionALCARECOPt5)
dtCalibOfflineSelectionCosmicsPt5 = cms.Sequence(offlineSelectionCosmicsPt5)
dtCalibOfflineSelectionHIPt5 = cms.Sequence(offlineSelectionHIPt5)
dtCalibOfflineSelectionHIALCARECOPt5 = cms.Sequence(offlineSelectionHIALCARECOPt5)
dtCalibOfflineSelectionHIRAWPt5 = cms.Sequence(offlineSelectionHIRAWPt5)

dtCalibOfflineSelection = cms.Sequence(offlineSelection)
dtCalibOfflineSelectionALCARECO = cms.Sequence(offlineSelectionALCARECO)
dtCalibOfflineSelectionALCARECODtCalibTest = cms.Sequence(offlineSelectionALCARECODtCalibTest)
dtCalibOfflineSelectionCosmics = cms.Sequence(offlineSelectionCosmics)
dtCalibOfflineSelectionHI = cms.Sequence(offlineSelectionHI)
dtCalibOfflineSelectionHIALCARECO = cms.Sequence(offlineSelectionHIALCARECO)
dtCalibOfflineSelectionHIRAW = cms.Sequence(offlineSelectionHIRAW)
dtCalibOfflineSelectionTestEnables = cms.Sequence(offlineSelectionTestEnables)
