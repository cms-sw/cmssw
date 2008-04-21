import FWCore.ParameterSet.Config as cms

#Event content for PhiSymmetry alcareco stream
HLTAlcaRecoEcalPhiSymStreamFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 
        'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 
        'keep *_hltL1GtUnpack_*_*', 
        'keep *_hltGtDigis_*_*')
)
HLTAlcaRecoEcalPhiSymStreamRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 
        'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 
        'keep *_hltL1GtUnpack_*_*', 
        'keep *_hltGtDigis_*_*')
)
HLTAlcaRecoEcalPhiSymStreamAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)

