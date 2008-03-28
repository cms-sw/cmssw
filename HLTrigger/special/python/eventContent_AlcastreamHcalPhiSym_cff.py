import FWCore.ParameterSet.Config as cms

#Event content for HCAL PhiSymmetry alcareco stream
HLTAlcaRecoHcalPhiSymStreamFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_alCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_alCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_alCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')
)
HLTAlcaRecoHcalPhiSymStreamRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_alCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_alCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_alCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')
)
HLTAlcaRecoHcalPhiSymStreamAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)

