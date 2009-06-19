import FWCore.ParameterSet.Config as cms

pfAllMuons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("pileUpOnPFCandidates"),
    pdgId = cms.vint32( -13, 13)
)



