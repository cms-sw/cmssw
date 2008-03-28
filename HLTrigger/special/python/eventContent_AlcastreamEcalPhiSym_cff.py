import FWCore.ParameterSet.Config as cms

#Event content for PhiSymmetry alcareco stream
HLTAlcaRecoEcalPhiSymStreamFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_alCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_alCaPhiSymStream_phiSymEcalRecHitsEE_*')
)
HLTAlcaRecoEcalPhiSymStreamRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_alCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_alCaPhiSymStream_phiSymEcalRecHitsEE_*')
)
HLTAlcaRecoEcalPhiSymStreamAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)

