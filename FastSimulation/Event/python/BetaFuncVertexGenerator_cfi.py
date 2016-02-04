import FWCore.ParameterSet.Config as cms

myVertexGenerator = cms.PSet(
    # half-crossing beam angle
    Phi = cms.double(0.000142),
    BetaStar = cms.double(55.0),
    type = cms.string('BetaFunc'),
    Emittance = cms.double(5.03e-08),
    # angle of the crossing plane 0 degrees means XZ plane
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(7.55),
    Y0 = cms.double(0.0),
    # Units are cm and radians
    X0 = cms.double(0.05),
    Z0 = cms.double(0.0)
)

