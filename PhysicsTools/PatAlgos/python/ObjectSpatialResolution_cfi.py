# The following comments couldn't be translated into the new config version:

#
# Modules for smearing 4-vector's angles of Objects
#
# Parameters:
# - InputTag movedObject:
#   Specify object to smear. 
#   Object type should correspond to module (e.g. Electron in module ElectronSpatialResolution) 
# - bool useDefaultInitialResolutions :
#   Objects contain individual 4-vector resolutions (in terms of Et, theta/eta, phi).
#   Set this to "true" if these should be used to compute initial resolutions for the smearing.
# - bool usePolarTheta:
#   Switch to specify, if eta or rather theta is used for the polar angle smearing.
#   Set this to "true" to smear theta or to "false" to smear eta.
# - double initialResolutionPolar:
#   Initial resolution (in radiants for theta) to be used for the eta/theta smearing.
#   Is given as absolute value only, since factors make no sense for angles.
#   Overwritten, if 'useDefaultInitialResolution' is "true".
# - double worsenResolutionPolar:
#   Used to calculate the final resolution (after smearing) from the initial resolution.
#   The angle is smeared with a Gaussion of
#   mu    = angle and
#   sigma = sqrt(finalRes^2-iniRes^2).
# - bool worsenResolutionPolarByFactor:
#   Flags the usage mode of 'worsenResolutionPolar' (how the final resolution is calculated from the initial one)
# - double initialResolutionPhi, double worsenResolutionPhi and bool worsenResolutionPhiByFactor:
#   Accordingly...
#
# Examples:
#
# Remarks:
# - To switch off angular smearing, use (worsenResolutionPolar=0.) resp. (worsenResolutionPhi=0.)
# - All numeric values are protected from "meaninglessness" by the usage of absolute values only.
# - In the standard sequence at the bottom of this file, Taus are commented.
# - Keep polar smearing switched off for MET objects!!! ;-)
#
# Contact: volker.adler@cern.ch
#
# initialize random number generator

#  movedTaus &

import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    # need one initializer for each module defined below
    moduleSeeds = cms.PSet(
        movedJets = cms.untracked.uint32(61587),
        movedMuons = cms.untracked.uint32(17987),
        movedMETs = cms.untracked.uint32(3489766),
        movedElectrons = cms.untracked.uint32(897867),
        movedTaus = cms.untracked.uint32(38476)
    ),
    sourceSeed = cms.untracked.uint32(7893456)
)

movedElectrons = cms.EDFilter("ElectronSpatialResolution",
    worsenResolutionPolar = cms.double(2.0),
    worsenResolutionPhi = cms.double(1.0),
    worsenResolutionPolarByFactor = cms.bool(True),
    usePolarTheta = cms.bool(True),
    movedObject = cms.InputTag("selectedLayer1Electrons"),
    useDefaultInitialResolutions = cms.bool(False),
    worsenResolutionPhiByFactor = cms.bool(True),
    initialResolutionPhi = cms.double(0.0),
    initialResolutionPolar = cms.double(0.005)
)

movedMuons = cms.EDFilter("MuonSpatialResolution",
    worsenResolutionPolar = cms.double(2.0),
    worsenResolutionPhi = cms.double(1.0),
    worsenResolutionPolarByFactor = cms.bool(True),
    usePolarTheta = cms.bool(False),
    movedObject = cms.InputTag("selectedLayer1Muons"),
    useDefaultInitialResolutions = cms.bool(False),
    worsenResolutionPhiByFactor = cms.bool(True),
    initialResolutionPhi = cms.double(0.0),
    initialResolutionPolar = cms.double(0.005)
)

movedTaus = cms.EDFilter("TauSpatialResolution",
    worsenResolutionPolar = cms.double(1.0),
    worsenResolutionPhi = cms.double(1.0),
    worsenResolutionPolarByFactor = cms.bool(True),
    usePolarTheta = cms.bool(True),
    movedObject = cms.InputTag("selectedLayer1Taus"),
    useDefaultInitialResolutions = cms.bool(False),
    worsenResolutionPhiByFactor = cms.bool(True),
    initialResolutionPhi = cms.double(0.0),
    initialResolutionPolar = cms.double(0.0)
)

movedJets = cms.EDFilter("JetSpatialResolution",
    worsenResolutionPolar = cms.double(0.25),
    worsenResolutionPhi = cms.double(1.0),
    worsenResolutionPolarByFactor = cms.bool(False),
    usePolarTheta = cms.bool(True),
    movedObject = cms.InputTag("selectedLayer1Jets"),
    useDefaultInitialResolutions = cms.bool(False),
    worsenResolutionPhiByFactor = cms.bool(True),
    initialResolutionPhi = cms.double(0.0),
    initialResolutionPolar = cms.double(0.1)
)

movedMETs = cms.EDFilter("METSpatialResolution",
    worsenResolutionPolar = cms.double(1.0),
    worsenResolutionPhi = cms.double(1.6),
    worsenResolutionPolarByFactor = cms.bool(True),
    usePolarTheta = cms.bool(True),
    movedObject = cms.InputTag("selectedLayer1METs"),
    useDefaultInitialResolutions = cms.bool(False),
    worsenResolutionPhiByFactor = cms.bool(False),
    initialResolutionPhi = cms.double(0.8),
    initialResolutionPolar = cms.double(0.0)
)

# Standard sequence for all objects
movedObjects = cms.Sequence(movedElectrons+movedMuons+movedJets+movedMETs)

