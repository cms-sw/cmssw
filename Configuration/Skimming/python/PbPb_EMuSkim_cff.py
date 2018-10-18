import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltEMuHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltEMuHI.HLTPaths = ["HLT_HIL1Mu5Eta2p5_Ele20Gsf_v*"]
hltEMuHI.throw = False
hltEMuHI.andOr = True

# selection of valid vertex
primaryVertexFilterForEMu = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# single lepton selector                                                                                                                                               
electronSelectorForEMu = cms.EDFilter("GsfElectronRefSelector",
                                   src = cms.InputTag("gedGsfElectrons"),
                                   cut = cms.string("pt > 20 && abs(eta)<1.44")
                                   )

muonSelectorForEMu = cms.EDFilter("MuonSelector",
                                  src = cms.InputTag("muons"),
                                  cut = cms.string("(isTrackerMuon && isGlobalMuon) && pt > 20."),
                                  filter = cms.bool(True)
    )


# EMu skim sequence
emuSkimSequence = cms.Sequence(
    hltEMuHI *
    primaryVertexFilterForEMu *
    electronSelectorForEMu * 
    muonSelectorForEMu
)
