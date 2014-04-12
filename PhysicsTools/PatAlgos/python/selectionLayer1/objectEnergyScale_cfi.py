import FWCore.ParameterSet.Config as cms

##
## Modules for shifting and smearing 4-vectors' energies of Objects
##
## Parameters:
## - InputTag scaledObject: 
##   Specify objects to shift & smear.
##   Object type should correspond to module (e.g. Electron in module ElectronEnergyScale) 
## - bool fixMass:
##   Set to "true", if mass should not be smeared, e.g. for leptons with known masses.
##   IMPORTANT: Don't trust this feature at the moment !!! (s. remarks below)
## - double shiftFactor:
##   Scales the 4-vector with this fixed value.
##   E.g. to have all muon energies increased by 12.5%, set "1.125" here.
##   Default is "1.", which has -- of course -- no effect
## - bool useDefaultInitialResolution:
##   Objects contain individual 4-vector resolutions (in terms of Et, theta/eta, phi).
##   Set this to "true" if these should be used to compute initial resolutions for the smearing.
## - double initialResolution:
##   Initial resolution to be used for the energy smearing.
##   Can be an absolute value (in GeV) or a factor giving the fraction of the smeared energy.
##   Overwritten, if 'useDefaultInitialResolution' is "true".
## - bool initialResolutionByFraction:
##   Flags the usage mode of 'initialResolution'.
##   E.g. to set the initial resolution to 5% of the energy, set this to "true" and 'initialResolution' to "0.05".
##   To use a fixed resolution of 1GeV, set this to "false" and 'initialResolution' to "1.".
## - double worsenResolution:
##   Used to calculate the final resolution (after smearing) from the initial resolution.
##   Can be an absolute value (in GeV) or a factor.
##   The energy is smeared with a Gaussion of
##   mu    = energy and
##   sigma = sqrt(finalRes^2-iniRes^2)
##   with a cut-off at 0.
## - bool worsenResolutionByFactor:
##   Flags the usage mode of 'worsenResolution'.
##
## Examples:
## - smear Electron 4-vector (fixed mass) with a fixed initial resolution of 500MeV to as final resolution of 1.25GeV:
##  scaledElectrons = cms.module(ElectronEnergyScale) {
##    scaledObject                = cms.InputTag("selectedPatElectrons")
##    fixMass                     = cms.bool(True)
##    shiftFactor                 = cms.double(1.)
##    useDefaultInitialResolution = cms.bool(False)
##    initialResolution           = cms.double(0.5)
##    initialResolutionByFraction = cms.bool(False)   ## alternative:
##    worsenResolution            = cms.double(0.75)  ## 2.5
##    worsenResolutionByFactor    = cms.bool(False)   ## true
##   }
## - smear Muon 4-vector (fixed mass) with a initial resolution of 10% to as final resoltion of 20%:
##  scaledMuons = cms.module(MuonEnergyScale) {
##    scaledObject                = cms.InputTag("selectedPatMuons")
##    fixMass                     = cms.bool(True)
##    shiftFactor                 = cms.double(1.)
##    useDefaultInitialResolution = cms.bool(True)
##    initialResolution           = cms.double(0.1)
##    initialResolutionByFraction = cms.bool(True)
##    worsenResolution            = cms.double(2.)
##    worsenResolutionByFactor    = cms.bool(True)
##   }
## - smear Jet 4-vector to a final resoltion of 150% of the default initial resolution:
##  scaledJets = cms.module(JetEnergyScale) {
##    scaledObject                = cms.InputTag("selectedPatAK5CaloJets")
##    fixMass                     = cms.bool(False)  ## ===> no fixed mass for the jet
##    shiftFactor                 = cms.double(1.)
##    useDefaultInitialResolution = cms.bool(True)
##    initialResolution           = cms.double(0.05) ## ===> overwritten by "useDefaultInitialResolution = true"
##    initialResolutionByFraction = cms.bool(True)
##    worsenResolution            = cms.double(1.5)
##    worsenResolutionByFactor    = cms.bool(True)
##   }
##
## Remarks:
## - Due to the inclusion of the default initial resolutions, these modules are limited to Objects for the moment.
## - The smearing takes care, that final resolutions do not become smaller than initial resolutions.
##   E.g. if (worsenResolution=0.6 && worsenResolutionByFactor=true) is set, it is assumed that the final resolution
##   should be 40% worse than the initial resolution. So, 'worsenResolution' is set to 1.4 internally.
##   (Analogously for (worsenResolution<0. && worsenResolutionByFactor=false).)
## - To switch off energy shifting, use (shiftFactor=1.).
## - To switch off energy smearing, use (worsenResolution=0. && worsenResolutionByFactor=false) or
##   (worsenResolution=1. && worsenResolutionByFactor=true).
## - In the standard sequence at the bottom of this file, Taus are commented.
## - (fixMass=true) isn't reliable so far :-(
##   The input provided by class Particle is not yet understood (Negative mass for positive mass^2 in contradiction to ROOT::TLorentzVector).
##
## Contact: volker.adler@cern.ch
##


## initialize random number generator
RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    scaledElectrons = cms.PSet(
        initialSeed = cms.untracked.uint32(897867),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    scaledMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(17987),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    theSource = cms.PSet(
        initialSeed = cms.untracked.uint32(7893456),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    scaledTaus = cms.PSet(
        initialSeed = cms.untracked.uint32(38476),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    scaledJets = cms.PSet(
        initialSeed = cms.untracked.uint32(61587),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    scaledMETs = cms.PSet(
        initialSeed = cms.untracked.uint32(3489766),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)

## Electrons
scaledElectrons = cms.EDFilter("ElectronEnergyScale",
 scaledObject                = cms.InputTag("selectedPatElectrons"),
 fixMass                     = cms.bool(False),
 shiftFactor                 = cms.double(1.),
 useDefaultInitialResolution = cms.bool(True),
 initialResolution           = cms.double(0.),
 initialResolutionByFraction = cms.bool(True),
 worsenResolution            = cms.double(1.),
 worsenResolutionByFactor    = cms.bool(True),
)

## Muons
scaledMuons = cms.EDFilter("MuonEnergyScale",
 scaledObject                = cms.InputTag("selectedPatMuons"),
 fixMass                     = cms.bool(False),
 shiftFactor                 = cms.double(1.),
 useDefaultInitialResolution = cms.bool(True),
 initialResolution           = cms.double(0.),
 initialResolutionByFraction = cms.bool(True),
 worsenResolution            = cms.double(1.),
 worsenResolutionByFactor    = cms.bool(True),
)

## Taus
scaledTaus = cms.EDFilter("TauEnergyScale",
 scaledObject                = cms.InputTag("selectedPatTaus"),
 fixMass                     = cms.bool(False),
 shiftFactor                 = cms.double(1.),
 useDefaultInitialResolution = cms.bool(True),
 initialResolution           = cms.double(0.),
 initialResolutionByFraction = cms.bool(True),
 worsenResolution            = cms.double(1.),
 worsenResolutionByFactor    = cms.bool(True),
)

## Jets
scaledJets = cms.EDFilter("JetEnergyScale",
 scaledObject                = cms.InputTag("selectedPatJets"),
 fixMass                     = cms.bool(False),
 shiftFactor                 = cms.double(1.),
 useDefaultInitialResolution = cms.bool(True),
 initialResolution           = cms.double(0.),
 initialResolutionByFraction = cms.bool(True),
 worsenResolution            = cms.double(1.),
 worsenResolutionByFactor    = cms.bool(True),
)

## METs
scaledMETs = cms.EDFilter("METEnergyScale",
 scaledObject                = cms.InputTag("selectedPatMETs"),
 fixMass                     = cms.bool(False),
 shiftFactor                 = cms.double(1.),
 useDefaultInitialResolution = cms.bool(True),
 initialResolution           = cms.double(0.),
 initialResolutionByFraction = cms.bool(True),
 worsenResolution            = cms.double(1.),
 worsenResolutionByFactor    = cms.bool(True),
)

## Standard sequence for all Objects
scaledObjects = cms.Sequence(
    scaledElectrons + 
    scaledMuons + 
    scaledJets + 
#   scaledTaus +
    scaledMETs
)
