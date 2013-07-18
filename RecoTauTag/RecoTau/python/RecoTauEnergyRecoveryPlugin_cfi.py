import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts

pfTauEnergyRecoveryPlugin = cms.PSet(
    qualityCuts = PFTauQualityCuts,
    corrLevel = cms.uint32(3), # enable all corrections
    lev1PhiWindow = cms.double(0.50),
    lev1EtaWindow = cms.double(0.10)
)
