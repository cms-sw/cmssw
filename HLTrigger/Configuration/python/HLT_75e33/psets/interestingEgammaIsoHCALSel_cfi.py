import FWCore.ParameterSet.Config as cms

interestingEgammaIsoHCALSel = cms.PSet(
    maxDIEta = cms.int32(5),
    maxDIPhi = cms.int32(5),
    minEnergyHB = cms.double(0.1),
    minEnergyHEDefault = cms.double(0.2),
    minEnergyHEDepth1 = cms.double(0.1)
)