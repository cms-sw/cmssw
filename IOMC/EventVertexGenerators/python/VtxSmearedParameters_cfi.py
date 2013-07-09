import FWCore.ParameterSet.Config as cms

#
# All units are cm and radians
#
# UNITS:
#
# TimeOffset in nanoseconds
# spacial displacement in cm

# common parameters
VtxSmearedCommon = cms.PSet(
    src = cms.InputTag("generator"),
    readDB = cms.bool(False)
)
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
#############################################
# Beta functions smearing (pp 7+7 TeV)
#
# Values taken from LHC optics simulation V6.5:
# see http://proj-lhc-optics-web.web.cern.ch/proj-lhc-optics-web/V6.500/IR5.html
# alpha = angle of the crossing plane 0 degrees means XZ plane
# phi = half-crossing beam angle
#
# Emittance is the no normalized emittance in cm = normalized emittance/gamma (beta=1)
# 
# length variables are in cm
#

# 900 GeV collisions, transverse beam size = 293 microns 
Early900GeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1100.0),
    Emittance = cms.double(1.564e-06),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(7.4),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
#  2.2 TeV collisions, transverse beam size 188 microns
Early2p2TeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1100.0),
    Emittance = cms.double(6.4e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.5),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
#  7 TeV collisions, transverse beam size with betastar=  11m is 105 microns,
Early7TeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1100.0),
    Emittance = cms.double(2.0e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.2),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
#  7 TeV collisions, transverse beam size with betastar=  2m is  45 microns,
Nominal7TeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(200.0),
    Emittance = cms.double(2.0e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.2),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
# 900 GeV realistic 2010 collisions, transverse beam size is 200 microns
Realistic900GeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1000.0),
    Emittance = cms.double(8.34e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(6.17),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2452),
    Y0 = cms.double(0.3993),
    Z0 = cms.double(0.8222)
)
# 7 TeV realistic collisions, beamspot width ~28 microns - appropriate for 2nd half of Commissioning10
Realistic7TeVCollisionComm10VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(200.0),
    Emittance = cms.double(0.804e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.50),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# 7 TeV realistic collisions, beamspot width ~43 microns - appropriate for 2010A
Realistic7TeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(350.0),
    Emittance = cms.double(1.072e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(6.26),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# 7 TeV realistic collisions, beamspot width ~38 microns - appropriate for 2010B
Realistic7TeVCollision2010BVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(350.0),
    Emittance = cms.double(0.804e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.40),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# 7 TeV realistic collisions, updated for 2011
# normalized emittance 2.5 microns, transverse beam size is 32 microns
Realistic7TeV2011CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(150.0),
    Emittance = cms.double(0.67e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.22),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# HI realistic collisions, updated for 2011
# estimated beamspot width 31-35 microns
RealisticHI2011CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(100.0),
    Emittance = cms.double(2.04e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(7.06),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2245),
    Y0 = cms.double(0.4182),
    Z0 = cms.double(0.0847)
)
# 2.76 TeV estimated collisions, 11m beta*
# normalized emittance 2.5 microns, transverse beam size is 140 microns
Realistic2p76TeV2011CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1100.0),
    Emittance = cms.double(1.70e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.22),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# 2.76 TeV estimated collisions for 2013, 11m beta*
# sigmaZ set to 8 cm
Realistic2p76TeV2013CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1100.0),
    Emittance = cms.double(1.70e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(8.0),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# HI realistic pPb collisions, updated for 2013
# 
RealisticHIpPb2013CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(80.0),
    Emittance = cms.double(6.25e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(8.0),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# 7 TeV centered collisions with parameters for 2011
# normalized emittance 2.5 microns, transverse beam size is 32 microns
Centered7TeV2011CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(150.0),
    Emittance = cms.double(0.67e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.22),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0),
    Y0 = cms.double(0.0),
    Z0 = cms.double(0.0)
)
# 8 TeV realistic collisions, transverse beam width size is 20 microns
Realistic8TeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(70.0),
    Emittance = cms.double(0.586e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(6.16),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# 8 TeV realistic collisions, transverse beam width size is 20 microns, updated for observed SigmaZ
Realistic8TeV2012CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(70.0),
    Emittance = cms.double(0.586e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.8),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.2440),
    Y0 = cms.double(0.3929),
    Z0 = cms.double(0.4145)
)
# 10 TeV collisions, transverse beam size = 46 microns
Early10TeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(300.0),
    Emittance = cms.double(1.406e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
# Test offset
Early10TeVX322Y100VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(300.0),
    Emittance = cms.double(1.406e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0100),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
# Test offset
Early10TeVX322Y250VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(300.0),
    Emittance = cms.double(1.406e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0250),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
# Test offset
Early10TeVX322Y500VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(300.0),
    Emittance = cms.double(1.406e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.0500),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
# Test offset
Early10TeVX322Y1000VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(300.0),
    Emittance = cms.double(1.406e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.1),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
# Test offset
Early10TeVX322Y5000VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(300.0),
    Emittance = cms.double(1.406e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.5),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)
# Test offset
Early10TeVX322Y10000VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(300.0),
    Emittance = cms.double(1.406e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(1.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)

EarlyCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(200.0),
    Emittance = cms.double(1.006e-07),
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
    Emittance = cms.double(1.006e-07),
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
    Emittance = cms.double(1.006e-07),
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
    Emittance = cms.double(1.006e-07),
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
    Emittance = cms.double(1.006e-07),
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
    Emittance = cms.double(1.006e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    Y0 = cms.double(0.025),
    X0 = cms.double(0.2),
    Z0 = cms.double(0.0)
)


