import FWCore.ParameterSet.Config as cms

# HLT PU-subtracted AK4 Calo. Jet trigger, highest threshold w/ full eta coverage
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltPbPbHighPtJet = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltPbPbHighPtJet.HLTPaths = ["HLT_HIPuAK4CaloJet120Eta5p1_v*"]
hltPbPbHighPtJet.throw = False
hltPbPbHighPtJet.andOr = True

# At reco, add filters kicking pT up to 300 GeV
jetPtCut = 300
jetEtaCut = 2.4
pfJetSelector = cms.EDFilter(
    "EtaPtMinCandViewSelector",
    src = cms.InputTag("akCs4PFJets"),
    ptMin = cms.double(jetPtCut),
    etaMin = cms.double(-jetEtaCut),
    etaMax = cms.double(jetEtaCut)
)
pfJetFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("pfJetSelector"),
    minNumber = cms.uint32(1)
)

# PbPb High-pT Jets skim sequence
pbpbHighPtJetSkimSequence = cms.Sequence(hltPbPbHighPtJet * pfJetSelector * pfJetFilter)
