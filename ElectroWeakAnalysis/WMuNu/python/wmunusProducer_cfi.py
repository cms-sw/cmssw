import FWCore.ParameterSet.Config as cms

# Producers --> Create one collection of WMuNus per met type

pfMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("pfMet"),
      OnlyHighestPtCandidate = cms.untracked.bool(True) # Only 1 Candidate saved in the event
)

tcMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("tcMet"), 
      OnlyHighestPtCandidate = cms.untracked.bool(True) 

)

corMetWMuNus = cms.EDProducer("WMuNuProducer",
      # Input collections ->
      MuonTag = cms.untracked.InputTag("muons"),
      METTag = cms.untracked.InputTag("corMetGlobalMuons"),
      OnlyHighestPtCandidate = cms.untracked.bool(True)
)

allWMuNus = cms.Sequence(corMetWMuNus
                   *tcMetWMuNus
                   *pfMetWMuNus
                 )

