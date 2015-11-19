import FWCore.ParameterSet.Config as cms

# HLT jet trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltJetHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltJetHI.HLTPaths = ["HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v*"]
hltJetHI.throw = False
hltJetHI.andOr = True

# selection of valid vertex
primaryVertexFilterForBJets = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# jet energy correction (L2+L3)
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
akVs4PFJetsL2L3 = cms.EDProducer('PFJetCorrectionProducer',
    src = cms.InputTag('akVs4PFJets'),
    correctors = cms.vstring('ak4PFL2L3')
    )

hiPtBJet = cms.EDFilter("PFJetSelector",
    src = cms.InputTag("akVs4PFJetsL2L3"),
    cut = cms.string("pt > 110")
    )

# dijet skim sequence
bJetSkimSequence = cms.Sequence(hltJetHI
                                 * primaryVertexFilterForBJets
                                 * akVs4PFJetsL2L3
                                 * hiPtBJet
                                 )
