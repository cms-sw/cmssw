import FWCore.ParameterSet.Config as cms

# HLT jet trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltJetHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltJetHI.HLTPaths = ["HLT_HIJet50U"]
hltJetHI.throw = False
hltJetHI.andOr = True

# selection of valid vertex
primaryVertexFilterForJets = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# jet energy correction (L2+L3)
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
icPu5CaloJetsL2L3 = cms.EDProducer('CaloJetCorrectionProducer',
    src = cms.InputTag('iterativeConePu5CaloJets'),
    correctors = cms.vstring('ic5CaloL2L3')
    )

# leading jet E_T filter
jetEtFilter = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("icPu5CaloJetsL2L3"),
    etMin = cms.double(110.0),
    minNumber = cms.uint32(1)
    )

# Dijet requirement
leadingCaloJet = cms.EDFilter( "LargestEtCaloJetSelector",
    src = cms.InputTag( "icPu5CaloJetsL2L3" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 1 )
    )

goodLeadingJet = cms.EDFilter("CaloJetSelector",
    src = cms.InputTag("leadingCaloJet"),
    cut = cms.string("et > 130")
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

# dijet skim sequence
diJetSkimSequence = cms.Sequence(hltJetHI
                                 * primaryVertexFilterForJets
                                 * icPu5CaloJetsL2L3
                                 * jetEtFilter
                                 * leadingCaloJet
                                 * goodLeadingJet
                                 * goodSecondJet
                                 * backToBackDijets
                                 * dijetFilter
                                 )
