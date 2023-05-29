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
    src = cms.InputTag("generator", "unsmeared"),
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
# Gaussian smearing
GaussVtxSigmaZ4cmSmearingParameters = cms.PSet(
    MeanX = cms.double(0.0),
    MeanY = cms.double(0.0),
    MeanZ = cms.double(0.0),
    SigmaY = cms.double(0.0015),
    SigmaX = cms.double(0.0015),
    SigmaZ = cms.double(4.0),
    TimeOffset = cms.double(0.0)
)
# Gaussian smearing
# Flat optics for Run3 - Low SigmaZ
# SigmaZ = 4.2 cm
# SigmaX = 11.8 um
# SigmaY = 5.5 um
# BS positions extracted from 2018B 3.8T data, run 316199, fill 6675 (from StreamExpressAlignment, HP BS):
# X0         =  0.09676  [cm]
# Y0         = -0.06245  [cm]
# Z0         = -0.292    [cm]
# BPIX absolute position (from https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod):
# X = 0.0859918 cm
# Y = -0.104172 cm
# Z = -0.327748 cm
Run3FlatOpticsGaussVtxSigmaZ4p2cmSmearingParameters = cms.PSet(
    MeanX = cms.double(0.0107682),
    MeanY = cms.double(0.041722),
    MeanZ = cms.double(0.035748),
    SigmaY = cms.double(0.00055),
    SigmaX = cms.double(0.00118),
    SigmaZ = cms.double(4.2),
    TimeOffset = cms.double(0.0)
)
# Gaussian smearing
# Flat optics for Run3 - High SigmaZ
# SigmaZ = 5.3 cm
# SigmaX = 15 um
# SigmaY = 13 um
# BS positions extracted from 2018B 3.8T data, run 316199, fill 6675 (from StreamExpressAlignment, HP BS):
# X0         =  0.09676  [cm]
# Y0         = -0.06245  [cm]
# Z0         = -0.292    [cm]
# BPIX absolute position (from https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod):
# X = 0.0859918 cm
# Y = -0.104172 cm
# Z = -0.327748 cm
Run3FlatOpticsGaussVtxSigmaZ5p3cmSmearingParameters = cms.PSet(
    MeanX = cms.double(0.0107682),
    MeanY = cms.double(0.041722),
    MeanZ = cms.double(0.035748),
    SigmaY = cms.double(0.0013),
    SigmaX = cms.double(0.0015),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0)
)

# Flat Smearing
# Important note: flat independent distributions in Z and T are not correct for physics production
# In reality, if two flat beams interact the real distribution will not be flat with independent Z and T
# but Z and T will be correlated, as example in GaussEvtVtxGenerator.
# Can restore correlation via MinT += (MinZ - MaxZ)/2 and MaxT += (MaxZ - MinZ)/2
# in [ns] units (recall c_light = 29.98cm/ns)
FlatVtxSmearingParameters = cms.PSet(
    MaxZ = cms.double(5.3),
    MaxX = cms.double(0.0015),
    MaxY = cms.double(0.0015),
    MinX = cms.double(-0.0015),
    MinY = cms.double(-0.0015),
    MinZ = cms.double(-5.3),
    MaxT = cms.double(0.177),
    MinT = cms.double(-0.177)
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
NominalCollision2015VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(65.0),
    Emittance = cms.double(5.411e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0322),
    Y0 = cms.double(0.0),
    Z0 = cms.double(0.0)
)
ZeroTeslaRun247324CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(80.0),
    Emittance = cms.double(1.070e-5),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.125),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.08621),
    Y0 = cms.double(0.1657),
    Z0 = cms.double(-1.688)
)

# From 2015A 0T data
# Centroid absolute positions extracted from fills:
# X = 0.059395  cm
# Y = 0.099686  cm
# Z = -1.722240 cm
#
# BPIX absolute position extracted from first collision alignment:
# X = -0.0259503 cm
# Y = -0.07004   cm
# Z = -0.498917  cm
Realistic50ns13TeVCollisionZeroTeslaVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(65.0),
    Emittance = cms.double(5.411e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.08533),
    Y0 = cms.double(0.16973),
    Z0 = cms.double(-1.2230)
)

# From 2015B 3.8T data
# Centroid absolute positions extracted from fill 4008:
# X =  0.07798 cm
# Y =  0.09714 cm
# Z = -1.610   cm
#
# BPIX absolute position extracted from PCL-like alignment run after magnet ramp-up:
# X = -0.026837  cm
# Y = -0.0715252 cm
# Z = -0.511453  cm
Realistic50ns13TeVCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(65.0),
    Emittance = cms.double(5.411e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.10482),
    Y0 = cms.double(0.16867),
    Z0 = cms.double(-1.0985)
)

# From 2015B 3.8T data, beta*=90m (700 bunches fills)
# Centroid absolute positions extracted from 700 bunches fills 4499-4511:
# X = 0.068357 cm
# Y = 0.109159 cm
# Z = 0.131811 cm
#
# BPIX absolute position extracted from Prompt Reco alignment of run 259352
# X = -0.041651 cm
# Y = -0.199279 cm
# Z = -0.565093 cm
#
# Emittance has been calculated to match a BeamWidht of O(10um) with: https://lpc.web.cern.ch/lumi2.html
#
Realistic100ns13TeVCollisionBetaStar90mVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(9121.0),
    Emittance = cms.double(0.12e-7),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.9),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.11000),
    Y0 = cms.double(0.30844),
    Z0 = cms.double(0.69690)
)

# From 2015B 3.8T data, beta*=90m (42/240 bunches fills)
# Centroid absolute positions extracted from 42/240 bunches fills 4495-4496:
# X = 0.064925 cm
# Y = 0.112761 cm
# Z = 0.170413 cm
#
# BPIX absolute position extracted from Prompt Reco alignment of run 259202
# X = -0.041651 cm
# Y = -0.199279 cm
# Z = -0.565093 cm
#
# Emittance has been calculated to match a BeamWidht of O(10um) with: https://lpc.web.cern.ch/lumi2.html
#
Realistic100ns13TeVCollisionBetaStar90mLowBunchesVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(9121.0),
    Emittance = cms.double(0.12e-7),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.24),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.10658),
    Y0 = cms.double(0.31204),
    Z0 = cms.double(0.735506)
)

# From 2016B 3.8T data
# BS parameters extracted from fills 4895 - 4935:
# X0         = 0.064870 +/- 4.9575E-07 [cm]
# Y0         = 0.093639 +/- 4.9233E-07 [cm]
# Z0         = 0.420085 +/- 4.1102E-04 [cm]
# sigmaZ0    = 3.645533 +/- 2.9064E-04 [cm]
#
# From LHC calculator, emittance is 4.906e-8 cm
# https://lpc.web.cern.ch/lpc/lumi2.html
#
# BPIX absolute position:
# X = -0.0267572 cm
# Y = -0.0759102 cm
# Z = -0.511428  cm
Realistic25ns13TeV2016CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(40.0),
    Emittance = cms.double(4.906e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.65),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.09163),
    Y0 = cms.double(0.16955),
    Z0 = cms.double(0.9315 )
)

# From 2017A 3.8T data
# BS parameters extracted from run 295463 (from offline DQM, i.e. PCL):
# X0         =  0.08497  [cm]
# Y0         = -0.03976  [cm]
# Z0         =  1.6      [cm] ==> 0.5 adjusted after cogging tuning by LHC, see  https://hypernews.cern.ch/HyperNews/CMS/get/beamspot/159/1.html
# sigmaZ0    =  3.5      [cm]
#
# From LHC calculator, emittance is 3.319e-8 cm
# https://lpc.web.cern.ch/lpc/lumi2.html
#
# BPIX absolute position (https://hypernews.cern.ch/HyperNews/CMS/get/tif-alignment/657/1/1.html):
# X = 0.109725 cm
# Y = -0.108993 cm
# Z = -0.32054  cm
Realistic25ns13TeVEarly2017CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(40.0),
    Emittance = cms.double(3.319e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.5),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(-0.024755),
    Y0 = cms.double(0.069233 ),
    Z0 = cms.double(0.82054  )
)

# Beam spot extracted from data for 2017 pp run @ 5 TeV
Realistic5TeVppCollision2017VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(311),
    Emittance = cms.double(3.8e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.82),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(-0.0228),
    Y0 = cms.double(0.0795),
    Z0 = cms.double(0.619)
)

# Fixed Emittance (X2) in Beam spot extracted from data for 2017 pp run @ 5 TeV
Fixed_EmitRealistic5TeVppCollision2017VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(311),
    Emittance = cms.double(7.6e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.82),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(-0.0228),
    Y0 = cms.double(0.0795),
    Z0 = cms.double(0.619)
)


# From 2018B 3.8T data
# BS parameters extracted from run 316199, fill 6675 (from StreamExpressAlignment, HP BS):
# X0         =  0.09676  [cm]
# Y0         = -0.06245  [cm]
# Z0         = -0.292    [cm]
# sigmaZ0    =  3.5      [cm] => mean sigmaZ0 in this run is 3.2676
# BeamWidthX 0.0008050
# BeamWidthY 0.0006238
#
# From LHC calculator, emittance is 1.634e-8 cm
# https://lpc.web.cern.ch/lpc/lumi2.html
#
# BPIX absolute position (from https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod):
# X = 0.0859918 cm
# Y = -0.104172 cm
# Z = -0.327748 cm
Realistic25ns13TeVEarly2018CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(30.0),
    Emittance = cms.double(1.634e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.5),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0107682),
    Y0 = cms.double(0.041722 ),
    Z0 = cms.double(0.035748 )
)

# Run3 possible beam parameters
# Round optics - Low SigmaZ = 3.4 cm
# From 2018B 3.8T data
# BS parameters extracted from run 316199, fill 6675 (from StreamExpressAlignment, HP BS):
# X0         =  0.09676  [cm]
# Y0         = -0.06245  [cm]
# Z0         = -0.292    [cm]
# sigmaZ0    =  3.2676   [cm]
# BeamWidthX 0.0008050
# BeamWidthY 0.0006238
#
# set SigmaZ0 = 3.4 [cm]
# set BeamWidthX = BeamWidthY = 11.5 [um]
# set beta* = 28 cm
# energy = 13 TeV
# From LHC calculator, emittance is 4.762e-8 cm
# https://lpc.web.cern.ch/lpc/lumi2.html
#
# BPIX absolute position (from https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod):
# X = 0.0859918 cm
# Y = -0.104172 cm
# Z = -0.327748 cm
Run3RoundOptics25ns13TeVLowSigmaZVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(28.0),
    Emittance = cms.double(4.762e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.4),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0107682),
    Y0 = cms.double(0.041722 ),
    Z0 = cms.double(0.035748 )
)

# Run3 possible beam parameters
# Round optics - High SigmaZ = 5.7 cm
# From 2018B 3.8T data
# BS parameters extracted from run 316199, fill 6675 (from StreamExpressAlignment, HP BS):
# X0         =  0.09676  [cm]
# Y0         = -0.06245  [cm]
# Z0         = -0.292    [cm]
# sigmaZ0    =  3.2676   [cm]
# BeamWidthX 0.0008050
# BeamWidthY 0.0006238
#
# set SigmaZ0 = 5.7 [cm]
# set BeamWidthX = BeamWidthY = 11.5 [um]
# set beta* = 28 cm
# energy = 13 TeV
# From LHC calculator, emittance is 4.762e-8 cm
# https://lpc.web.cern.ch/lpc/lumi2.html
#
# BPIX absolute position (from https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod):
# X = 0.0859918 cm
# Y = -0.104172 cm
# Z = -0.327748 cm
Run3RoundOptics25ns13TeVHighSigmaZVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(28.0),
    Emittance = cms.double(4.762e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.7),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0107682),
    Y0 = cms.double(0.041722 ),
    Z0 = cms.double(0.035748 )
)

# From Run3 PilotBeams 2021 3.8T data
# BS parameters extracted from run 346512, fill 7531 (from ExpressPhysics FEVT, Legacy BS):
# X0         =  0.174282 [cm]
# Y0         = -0.187132 [cm]
# Z0         =  0.167616 [cm]
# sigmaZ0    =  6.80728  [cm]
# BeamWidthX 0.0142174
# BeamWidthY 0.0150789
#
# set SigmaZ0 = 6.8 [cm]
# set BeamWidthX = BeamWidthY = 150.0 [um]
# set beta* = 1100 cm
# energy = 900 GeV
# From LHC calculator, emittance is 4.762e-8 cm
# https://lpc.web.cern.ch/lpc/lumi2.html
#
# BPIX absolute position (https://twiki.cern.ch/twiki/bin/view/CMS/TkAlignmentPixelPosition?rev=40#2021):
# X =  0.06076 cm
# Y = -0.14702 cm
# Z = -0.25616 cm
Realistic25ns900GeV2021CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1100.0),
    Emittance = cms.double(4.169e-7),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(6.8),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.113522),
    Y0 = cms.double(-0.040112),
    Z0 = cms.double(0.423776)
)

# From first Run 3 data at 13.6 TeV and 3.8T
# BS parameters extracted from run 355100, fill 7920:
# X0         =  0.172394 [cm]
# Y0         = -0.180946 [cm]
# Z0         =  0.94181  [cm]
# sigmaZ0    =  3.81941  [cm]
# BeamWidthX = 0.0008772 [cm]
# BeamWidthY = 0.0010078 [cm]
#
# set SigmaZ0 = 3.8 [cm]
# set BeamWidthX = BeamWidthY = 10.0 [um]
# set beta* = 30 cm
# energy = 13.6 TeV
# From LHC calculator, emittance is 6.621e-8 cm
# https://lpc.web.cern.ch/lumiCalc.html
#
# BPIX absolute position (https://twiki.cern.ch/twiki/bin/view/CMS/TkAlignmentPixelPosition?rev=45#Collisions_at_s_13_6_TeV):
# X =  0.0717651 cm
# Y = -0.165951  cm
# Z = -0.356345  cm
Realistic25ns13p6TeVEarly2022CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(30.0),
    Emittance = cms.double(6.621e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.8),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.100629),
    Y0 = cms.double(-0.014995),
    Z0 = cms.double(1.298155)
)

# BS parameters extracted from run 360459, Fill 8274:
# X0         =  0.1742 [cm]
# Y0         = -0.1831 [cm]
# Z0         = -0.2531 [cm]
# sigmaZ0    =  3.4019  [cm]
# BeamWidthX = 0.0007519 [cm]
# BeamWidthY = 0.0008636 [cm]
#
# set SigmaZ0 = 3.4 [cm]
# set BeamWidthX = BeamWidthY = 8.0 [um]
# set beta* = 30 cm
# energy = 13.6 TeV
# From LHC calculator, emittance is 4.276-8 cm
# https://lpc.web.cern.ch/lumiCalc.html
#
# BPIX absolute position:
# X =  0.0714025 cm
# Y = -0.166338  cm
# Z = -0.354856  cm
Realistic25ns13p6TeVEOY2022CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(30.0),
    Emittance = cms.double(4.276e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.4),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.1027975),
    Y0 = cms.double(-0.016762),
    Z0 = cms.double(0.101756)
)

# BS parameters extracted averaging Fills 8728-8750 (2023C):
# X0         =  0.117154 [cm]
# Y0         = -0.186556 [cm]
# Z0         = -0.431777 [cm]
# sigmaZ0    =  3.599 cm [cm]
# BeamWidthX = 0.0007333 [cm]
# BeamWidthY = 0.0008046 [cm]
#
# set SigmaZ0 = 3.6 [cm]
# set BeamWidthX = BeamWidthY = 7.7 [um]
# set beta* = 30 cm
# energy = 13.6 TeV
# From LHC calculator, emittance is 3.931e-8 cm
# https://lpc.web.cern.ch/lumiCalc.html
#
# BPIX absolute position (from Runs 367094-367589):
# X =  0.0713008 cm
# Y = -0.169590  cm
# Z = -0.356785  cm
Realistic25ns13p6TeVEarly2023CollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(30.0),
    Emittance = cms.double(3.931e-8),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(3.6),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0458532),
    Y0 = cms.double(-0.016966),
    Z0 = cms.double(-0.074992)
)

# Test HF offset
ShiftedCollision2015VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(65.0),
    Emittance = cms.double(5.411e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(1.0),
    Y0 = cms.double(0.0),
    Z0 = cms.double(0.0)
)
Shifted5mmCollision2015VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(65.0),
    Emittance = cms.double(5.411e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.5),
    Y0 = cms.double(0.0),
    Z0 = cms.double(0.0)
)
Shifted15mmCollision2015VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(65.0),
    Emittance = cms.double(5.411e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.3),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(1.5),
    Y0 = cms.double(0.0),
    Z0 = cms.double(0.0)
)

# Estimate for 2015 PbPb collisions, based on feedback from accelerator
# Beamspot centroid shifted to match pp expectation for 2015
NominalHICollision2015VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(60.0),
    Emittance = cms.double(1.70e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(7.06),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0322),
    Y0 = cms.double(0.),
    Z0 = cms.double(0.)
)

# updated numbers based on beamspot fits to 2015 PbPb data
# Later found to be incorrect, see following entry
# RealisticHICollision2015VtxSmearingParameters = cms.PSet(
#    Phi = cms.double(0.0),
#    BetaStar = cms.double(60.0),
#    Emittance = cms.double(1.70e-07),
#    Alpha = cms.double(0.0),
#    SigmaZ = cms.double(5.2278),
#    TimeOffset = cms.double(0.0),
#    X0 = cms.double(0.1025),
#    Y0 = cms.double(0.1654),
#    Z0 = cms.double(3.2528)
#)
# updated numbers for 2015 PbPb data with Z centroid from fixed beamspot fits
# See discussion here https://hypernews.cern.ch/HyperNews/CMS/get/hi-general/3968.html
# See plot of difference here: https://www.dropbox.com/s/tsnkgvvpkdqjtyq/vzDataMCOverlay_c_20170420.pdf?dl=0
#
RealisticHICollisionFixZ2015VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(60.0),
    Emittance = cms.double(1.70e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.2278),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.1025),
    Y0 = cms.double(0.1654),
    Z0 = cms.double(0.771)
)
# Numbers based on beamspot fits to 2017 XeXe data
# Documentation here: https://twiki.cern.ch/twiki/pub/CMS/XeXeRereco/IanLRU_AlCaTkAlBS_20171130_approvedByLucaSara.pdf
RealisticXeXeCollision2017VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(30.0),
    Emittance = cms.double(4.33e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.64),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(-0.026),
    Y0 = cms.double(0.081),
    Z0 = cms.double(0.645)
)

# From fit to 5 TeV PbPb data
# From 2018 PbPb  data
# BS parameters extracted from run 327211, Fill 7471 (from StreamExpressAlignment, HP, BS):
# X0         =  0.09443  [cm]
# Y0         = -0.06377  [cm]
# Z0         =  0.58067  [cm]
# sigmaZ0    =  4.969    [cm]
# BeamWidthX 0.0014392
# BeamWidthY 0.0011545
#
# Emittance is 3.36e-8 cm. Calculated by  ((BeamWidthX + BeamWidth)/2)^2/BetaStar
#
# BPIX absolute position (from https://cms-conddb.cern.ch/cmsDbBrowser/payload_inspector/Prod):
# X =  0.084335 cm
# Y = -0.110381 cm
# Z = -0.321904 cm
RealisticPbPbCollision2018VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(50),
    Emittance = cms.double(3.36e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.97),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.010),
    Y0 = cms.double(0.047),
    Z0 = cms.double(0.903)
)

# Estimate for 2015 pp collisions at 5.02 TeV, based on feedback from accelerator:  beta* ~ 400cm, normalized emittance = 2.5 um, SigmaZ similar to RunIIWinter15GS
Nominal5TeVpp2015VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(400.0),
    Emittance = cms.double(1.0e-07),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.5),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.1044),
    Y0 = cms.double(0.1676),
    Z0 = cms.double(0.6707)
)

# From fit to 5 TeV pPb data
Realistic5TeVPACollision2016VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(1100.0),
    Emittance = cms.double(6.75e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(6.4891),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0889),
    Y0 = cms.double(0.1820),
    Z0 = cms.double(1.6066)
)

# From fit to 8 TeV pPb data
Realistic8TeVPACollision2016VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(60.0),
    Emittance = cms.double(6.75e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.6914),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.0836),
    Y0 = cms.double(0.1837),
    Z0 = cms.double(1.3577)
)

# Guess for 2022 PbPb beam conditions, which takes the 2018 PbPb beam width parameters from RealisticPbPbCollision2018VtxSmearingParameters with the current pp MC beam centroid from Realistic25ns13p6TeVEarly2022Collision
Nominal2022PbPbCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(50),
    Emittance = cms.double(3.36e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(4.97),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.100629),
    Y0 = cms.double(-0.014995),
    Z0 = cms.double(1.298155)
)

# From 2022 PbPb test data 362294
Realistic2022PbPbCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(50),
    Emittance = cms.double(3.36e-08),
    Alpha = cms.double(0.0),
    SigmaZ = cms.double(5.01265),
    TimeOffset = cms.double(0.0),
    X0 = cms.double(0.1017599),
    Y0 = cms.double(-0.015602),
    Z0 = cms.double(0.131175)
)

# Parameters for HL-LHC operation at 13TeV
HLLHCVtxSmearingParameters = cms.PSet(
    MeanXIncm = cms.double(0.),
    MeanYIncm = cms.double(0.),
    MeanZIncm = cms.double(0.),
    TimeOffsetInns = cms.double(0.0),
    EprotonInGeV = cms.double(6500.0),
    CrossingAngleInurad = cms.double(510.0),
    CrabFrequencyInMHz = cms.double(400.0),
    RF800 = cms.bool(False),
    BetaCrossingPlaneInm = cms.double(0.20),
    BetaSeparationPlaneInm = cms.double(0.20),
    HorizontalEmittance = cms.double(2.5e-06),
    VerticalEmittance = cms.double(2.05e-06),
    BunchLengthInm = cms.double(0.090),
    CrabbingAngleCrossingInurad = cms.double(380.0),
    CrabbingAngleSeparationInurad = cms.double(0.0)
)

# Parameters for HL-LHC Crab-kissing operation 13 TeV
HLLHCCrabKissingVtxSmearingParameters = cms.PSet(
    MeanXIncm = cms.double(0.),
    MeanYIncm = cms.double(0.),
    MeanZIncm = cms.double(0.),
    TimeOffsetInns = cms.double(0.0),
    EprotonInGeV = cms.double(6500.0),
    HalfCrossingAngleInurad = cms.double(200.0),
    CrabAngleCrossingPlaneInurad = cms.double(200.0),
    CrabFrequencyCrossingPlaneInMHz = cms.double(400.0),
    NormalizedEmittanceCrossingPlaneInum = cms.double(2.5),
    BetaStarCrossingPlaneInm = cms.double(0.30),
    CrabAngleParallelPlaneInurad = cms.double(100.0),
    CrabFrequencyParallelPlaneInMHz = cms.double(400.0),
    NormalizedEmittanceParallelPlaneInum = cms.double(2.5),
    BetaStarParallelPlaneInm = cms.double(0.075),
    ZsizeInm = cms.double(0.15),
    BeamProfile=cms.string("Flat")
)
