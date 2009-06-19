import FWCore.ParameterSet.Config as cms

pfAllElectrons = cms.EDProducer("PdgIdPFCandidateSelector",
    src = cms.InputTag("pileUpOnPFCandidates"),
    pdgId = cms.vint32(11,-11)
)



