import FWCore.ParameterSet.Config as cms

MLPF_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep recoPFCandidates_mlpfProducer_*_*',
        )
)

