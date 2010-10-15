import FWCore.ParameterSet.Config as cms

# HLT jet trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltJetHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltJetHI.HLTPaths = ["HLT_HIJet35U"]
hltJetHI.throw = False
hltJetHI.andOr = True

# leading jet E_T filter
singleJet110 = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeConePu5CaloJets"),
    etMin = cms.double(110.0),
    minNumber = cms.uint32(1)
    )

# dijet E_T filter
diJet70 = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeConePu5CaloJets"),
    etMin = cms.double(70.0),
    minNumber = cms.uint32(2)
    )

# dijet skim sequence
diJetSkimSequence = cms.Sequence(hltJetHI
                                 * singleJet110
                                 # * diJet70
                                 )
