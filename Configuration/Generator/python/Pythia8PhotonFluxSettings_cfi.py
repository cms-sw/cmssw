import FWCore.ParameterSet.Config as cms

# configuration for photon flux in PbPb
PhotonFlux_PbPb = cms.PSet(
    beamTypeA = cms.int32(1000822080),
    beamTypeB = cms.int32(1000822080),
    radiusA = cms.untracked.double(6.636),
    radiusB = cms.untracked.double(6.636),
    zA = cms.untracked.int32(82),
    zB = cms.untracked.int32(82)  
)

# configuration for photon flux in OO
PhotonFlux_OO = cms.PSet(
    beamTypeA = cms.int32(80160),
    beamTypeB = cms.int32(80160),
    radiusA = cms.untracked.double(3.02),
    radiusB = cms.untracked.double(3.02),
    zA = cms.untracked.int32(8),
    zB = cms.untracked.int32(8)
)

# configuration for photon flux in NeNe
# upon consultation from PYTHIA authors, as a first guess, use same radius as OO
PhotonFlux_NeNe = cms.PSet(
    beamTypeA = cms.int32(1000100200),
    beamTypeB = cms.int32(1000100200),
    radiusA = cms.untracked.double(3.02),
    radiusB = cms.untracked.double(3.02),
    zA = cms.untracked.int32(10),
    zB = cms.untracked.int32(10)
)

# configuration for photon flux in XeXe
# radius from charged particle raa: https://arxiv.org/pdf/1809.00201.pdf
PhotonFlux_XeXe = cms.PSet(
    beamTypeA = cms.int32(5418),
    beamTypeB = cms.int32(5418),
    radiusA = cms.untracked.double(5.4),
    radiusB = cms.untracked.double(5.4),
    zA = cms.untracked.int32(54),
    zB = cms.untracked.int32(54)
)
