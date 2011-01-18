import FWCore.ParameterSet.Config as cms

# HLT jet trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltJetHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltJetHI.HLTPaths = ["HLT_HIJet35U"]
hltJetHI.throw = False
hltJetHI.andOr = True

# jet energy correction (L2+L3) ??
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

# dijet E_T filter
dijetEtFilter = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("icPu5CaloJetsL2L3"),
    etMin = cms.double(70.0),
    minNumber = cms.uint32(2)
    )

# dijet skim sequence
diJetSkimSequence = cms.Sequence(hltJetHI
                                 * icPu5CaloJetsL2L3
                                 * jetEtFilter
                                 # * dijetEtFilter
                                 )
