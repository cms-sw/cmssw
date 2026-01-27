import FWCore.ParameterSet.Config as cms

# primary vertex filter
primaryVertexFilterForPbPbEWSkim = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),
)

# lepton filter
goodMuonsForPbPbEWSkim = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("slimmedMuons"),
    cut = cms.string("pt >= 15.0 && passed('CutBasedIdLoose')")
)
goodElectronsForPbPbEWSkim = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("slimmedElectrons"),
    cut = cms.string("pt >= 15.0")
)
oneLeptonForPbPbEWSkim = cms.EDFilter("PATLeptonCountFilter",
    electronSource = cms.InputTag("goodElectronsForPbPbEWSkim"),
    muonSource     = cms.InputTag("goodMuonsForPbPbEWSkim"),
    tauSource      = cms.InputTag(""),
    countElectrons = cms.bool(True),
    countMuons     = cms.bool(True),
    countTaus      = cms.bool(False),
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(1000000),
)

# skim sequence
EWSkimSequence = cms.Sequence(
    primaryVertexFilterForPbPbEWSkim *
    goodMuonsForPbPbEWSkim *
    goodElectronsForPbPbEWSkim *
    oneLeptonForPbPbEWSkim
)

# skim content
from Configuration.EventContent.EventContent_cff import MINIAODEventContent
EWSkimContent = MINIAODEventContent.clone()
EWSkimContent.outputCommands.append("drop *_*_*_SKIM")
