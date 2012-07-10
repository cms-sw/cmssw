

from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
es_prefer_l1GtTriggerMaskAlgoTrig = cms.ESPrefer("L1GtTriggerMaskAlgoTrigTrivialProducer","l1GtTriggerMaskAlgoTrig")
es_prefer_l1GtTriggerMaskTechTrig = cms.ESPrefer("L1GtTriggerMaskTechTrigTrivialProducer","l1GtTriggerMaskTechTrig")
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *

# Good coll. --> '0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))'
l1tech = hltLevel1GTSeed.clone()
l1tech.L1TechTriggerSeeding = cms.bool(True)

bptx =l1tech.clone()
bptx.L1SeedsLogicalExpression = cms.string('0')

bscAnd = l1tech.clone()
bscAnd.L1SeedsLogicalExpression = cms.string('40 OR 41')

beamHaloVeto = l1tech.clone()
beamHaloVeto.L1SeedsLogicalExpression = cms.string('NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')

#l1Coll = cms.Sequence(bptx + beamHaloVeto)
l1Coll = cms.Sequence(bptx)
l1CollBscAnd = cms.Sequence(bptx + bscAnd + beamHaloVeto)

primaryVertexFilter = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"),
    filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
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

goodMuons = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('(isGlobalMuon = 1 | isTrackerMuon = 1) & abs(eta) < 1.2 & pt > 5.0'),
    filter = cms.bool(True) 
)
muonSelection = cms.Sequence(goodMuons)

offlineSelection = cms.Sequence(scrapingEvtFilter + primaryVertexFilter + muonSelection)
offlineSelectionALCARECO = cms.Sequence(muonSelection)
offlineSelectionCosmics = cms.Sequence(muonSelection)

dtCalibOfflineSelection = cms.Sequence(offlineSelection)
dtCalibOfflineSelectionALCARECO = cms.Sequence(offlineSelectionALCARECO)
dtCalibOfflineSelectionCosmics = cms.Sequence(offlineSelectionCosmics)
