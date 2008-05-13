import FWCore.ParameterSet.Config as cms

#
# All units are cm and radians
#
# UNITS:
#
# TimeOffset in nanoseconds
# spacial displacement in cm
# Gaussian smearing
GaussVtxSmearingParameters = cms.PSet(
    MeanX = cms.double(0.0),
    MeanY = cms.double(0.0),
    MeanZ = cms.double(0.0),
    SigmaY = cms.double(0.0015),
    SigmaX = cms.double(0.0015),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0)
)
# Flat Smearing
FlatVtxSmearingParameters = cms.PSet(
    MaxZ = cms.double(5.3),
    MaxX = cms.double(0.0015),
    MaxY = cms.double(0.0015),
    MinX = cms.double(-0.0015),
    MinY = cms.double(-0.0015),
    MinZ = cms.double(-5.3),
    TimeOffset = cms.double(0.0)
)
# Beta functions smearing (pp 7+7 TeV)
#
# Values taken from LHC optics simulation V6.5:
# see http://proj-lhc-optics-web.web.cern.ch/proj-lhc-optics-web/V6.500/IR5.html
# alpha = angle of the crossing plane 0 degrees means XZ plane
# phi = half-crossing beam angle
#
# length variables are in cm
#
# 10 TeV collisions, beam size = 46 microns
Early10TeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(300.0),
    Emittance = cms.double(7.03e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
EarlyCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(200.0),
    Emittance = cms.double(5.03e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
NominalCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.000142),
    BetaStar = cms.double(55.0),
    Emittance = cms.double(5.03e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0),
    X0 = cms.double(0.05),
    Z0 = cms.double(0.0)
)
NominalCollision1VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(55.0),
    Emittance = cms.double(5.03e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.025),
    X0 = cms.double(0.05),
    Z0 = cms.double(0.0)
)
NominalCollision2VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.000142),
    BetaStar = cms.double(55.0),
    Emittance = cms.double(5.03e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.025),
    X0 = cms.double(0.05),
    Z0 = cms.double(0.0)
)
NominalCollision3VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(55.0),
    Emittance = cms.double(5.03e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.025),
    X0 = cms.double(0.1),
    Z0 = cms.double(0.0)
)
NominalCollision4VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(55.0),
    Emittance = cms.double(5.03e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.025),
    X0 = cms.double(0.2),
    Z0 = cms.double(0.0)
)


