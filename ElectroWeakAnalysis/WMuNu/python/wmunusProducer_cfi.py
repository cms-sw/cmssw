import FWCore.ParameterSet.Config as cms

# Producers --> Create one collection of WMuNus per met type

pfMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("pfMet")
)

tcMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("tcMet")
)

corMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("corMetGlobalMuons")
)

allWMuNus = cms.Sequence(corMetWMuNus
                   *tcMetWMuNus
                   *pfMetWMuNus
                 )

