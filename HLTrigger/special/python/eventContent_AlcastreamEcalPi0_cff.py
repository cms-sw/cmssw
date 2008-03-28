import FWCore.ParameterSet.Config as cms

#Event content for Pi0 alcareco stream
HLTAlcaRecoEcalPi0StreamFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*_pi0EcalRecHitsEB_*')
)
HLTAlcaRecoEcalPi0StreamRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*_pi0EcalRecHitsEB_*')
)
HLTAlcaRecoEcalPi0StreamAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)

