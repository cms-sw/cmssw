import FWCore.ParameterSet.Config as cms

pfTauEnergyAlgorithmPlugin = cms.PSet(
    dRaddNeutralHadron = cms.double(0.12), # CV: enabled adding PFNeutralHadrons
    minNeutralHadronEt = cms.double(50.),
    dRaddPhoton = cms.double(-1.), # CV: disabled adding PFGammas
    minGammaEt = cms.double(10.)
)
