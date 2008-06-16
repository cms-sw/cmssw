import FWCore.ParameterSet.Config as cms

#Event content for HCAL PhiSymmetry alcareco stream
HLTAlcaRecoHcalPhiSymStreamFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')
)
HLTAlcaRecoHcalPhiSymStreamRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')
)
HLTAlcaRecoHcalPhiSymStreamAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)

