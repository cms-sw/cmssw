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
import HLTrigger.HLTfilters.hltHighLevel_cfi

# jet trigger
hltJetHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltJetHI.HLTPaths = ["HLT_HIJet35U"]
hltJetHI.throw = False
hltJetHI.andOr = True

# photon trigger
hltPhotonHI = hltJetHI.clone()
hltPhotonHI.HLTPaths = ["HLT_HIPhoton15"]

# dimuon trigger
hltMuHI = hltJetHI.clone()
hltMuHI.HLTPaths = ["HLT_HIL1DoubleMuOpen"]

##### Analysis selections #####

# PAT jets
selectedPatJets = cms.EDFilter("PATJetSelector",
    src = cms.InputTag("patJets"),
    cut = cms.string("et > 40")
    )

countPatJets = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedPatJets")
    )

# reco jets and dijets

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
icPu5CaloJetsL2L3 = cms.EDProducer('CaloJetCorrectionProducer',
    src = cms.InputTag('iterativeConePu5CaloJets'),
    correctors = cms.vstring('ic5CaloL2L3')
    )

leadingCaloJet = cms.EDFilter( "LargestEtCaloJetSelector",
    src = cms.InputTag( "icPu5CaloJetsL2L3" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 1 )
    )

goodLeadingJet = cms.EDFilter("CaloJetSelector",
    src = cms.InputTag("leadingCaloJet"),
    cut = cms.string("et > 80")
    )

goodSecondJet = cms.EDFilter("CaloJetSelector",
    src = cms.InputTag("icPu5CaloJetsL2L3"),
    cut = cms.string("et > 50")
    )

backToBackDijets = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('abs(deltaPhi(daughter(0).phi,daughter(1).phi)) > 2.5'),
    decay = cms.string("goodLeadingJet goodSecondJet")
    )

dijetFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("backToBackDijets"),
    minNumber = cms.uint32(1)
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
    cut = cms.string('et > 10.0 && hadronicOverEm < 0.1 && r9 > 0.8')
)

photonFilter = cms.EDFilter("EtMinPhotonCountFilter",
    src = cms.InputTag("goodPhotons"),
    etMin = cms.double(40.0),
    minNumber = cms.uint32(1)
)

# Z -> ee
photonCombiner = cms.EDProducer("CandViewShallowCloneCombiner",
  checkCharge = cms.bool(False),
  cut = cms.string('60 < mass < 120'),
  decay = cms.string('goodPhotons goodPhotons')
)

photonPairCounter = cms.EDFilter("CandViewCountFilter",
  src = cms.InputTag("photonCombiner"),
  minNumber = cms.uint32(1)
)
