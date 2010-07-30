import FWCore.ParameterSet.Config as cms

######################################################
# A set of filters for heavy-ion analysis skimming:
#   (1) L1 algo and technical bits
#   (2) HLT paths
#   (3) Event selection
#   (4) Physics objects


##### L1 selections #####
from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
from L1Trigger.Skimmer.l1Filter_cfi import l1Filter

# tech bit 0 - BPTX coincidence
bptxAnd = hltLevel1GTSeed.clone(
    L1TechTriggerSeeding = cms.bool(True),
    L1SeedsLogicalExpression = cms.string('0')
    )

# tech bit 34 - BSC single-side - bits 36,37,38,39 - BSC beam halo veto
bscOr = hltLevel1GTSeed.clone(
    L1TechTriggerSeeding = cms.bool(True),
    L1SeedsLogicalExpression = cms.string('(34) AND NOT (36 OR 37 OR 38 OR 39)')
    )

# algo bit 124 - BSC OR + BPTX OR
bscOrBptxOr = l1Filter.clone(
    algorithms = cms.vstring("L1_BscMinBiasOR_BptxPlusORMinus")
    )


##### HLT selections #####
import HLTrigger.HLTfilters.hltHighLevelDev_cfi

# jet trigger
hltJetHI = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone()
hltJetHI.HLTPaths = ["HLT_HIJet35U"]
hltJetHI.HLTPathsPrescales  = cms.vuint32(1)
hltJetHI.HLTOverallPrescale = cms.uint32(1)
hltJetHI.throw = False
hltJetHI.andOr = True

# photon trigger
hltPhotonHI = hltJetHI.clone()
hltPhotonHI.HLTPaths = ["HLT_Photon15"]

# dimuon trigger
hltMuHI = hltJetHI.clone()
hltMuHI.HLTPaths = ["HLT_L1DoubleMuOpen"]

##### Analysis selections #####

# jets
selectedPatJets = cms.EDFilter("PATJetSelector",
    src = cms.InputTag("patJets"),
    cut = cms.string("et > 40")
    )

countPatJets = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedPatJets")
    )

dijets = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeConePu5CaloJets"),
    etMin = cms.double(10.0),
    minNumber = cms.uint32(2)
    )

# muons
muonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isStandAloneMuon || isGlobalMuon) && pt > 1."),
    filter = cms.bool(True)
    )

muonFilter = cms.EDFilter("MuonCountFilter",
    src = cms.InputTag("muonSelector"),
    minNumber = cms.uint32(1)
    )

dimuonsMassCut = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string(' mass > 70 & mass < 120 & charge=0'),
    decay = cms.string("muonSelector@+ muonSelector@-")
    )

dimuonsMassCutFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonsMassCut"),
    minNumber = cms.uint32(1)
    )

# photons
goodPhotons = cms.EDFilter("PhotonSelector",
    src = cms.InputTag("photons"),
    cut = cms.string('et > 10.0')
)
