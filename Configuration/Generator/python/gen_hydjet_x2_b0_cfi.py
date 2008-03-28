# The following comments couldn't be translated into the new config version:

# This is a vector of ParameterSet names to be read, in this order
# They are  in the include files
# pythiaDefault HAS TO BE ALWAYS included
# If just the hard QCD dijets wanted, comment out the photons corresponding include and parameter set

# MSEL=1 (hard QCD dijets in) + hydjet needed initializations
import FWCore.ParameterSet.Config as cms

source = cms.Source("HydjetSource",
    shadowingSwitch = cms.int32(1),
    maxTransverseRapidity = cms.double(1.5),
    comEnergy = cms.double(5500.0),
    sigmaInelNN = cms.double(58.0),
    doRadiativeEnLoss = cms.bool(True),
    qgpInitialTemperature = cms.double(1.0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    aBeamTarget = cms.double(207.0),
    cFlag = cms.int32(0),
    hydjetMode = cms.string('kHydroQJets'),
    hadronFreezoutTemperature = cms.double(0.14),
    nMultiplicity = cms.int32(26000),
    qgpNumQuarkFlavor = cms.int32(0),
    doCollisionalEnLoss = cms.bool(True),
    bFixed = cms.double(0.0),
    maxLongitudinalRapidity = cms.double(4.0),
    bMin = cms.double(0.0),
    fracSoftMultiplicity = cms.double(1.0),
    maxEventsToPrint = cms.untracked.int32(0),
    bMax = cms.double(0.0),
    # initialize pythia
    PythiaParameters = cms.PSet(
        orcaSettings = cms.vstring('MSEL=0', 'PARP(67)=1.', 'PARP(82)=1.9', 'PARP(85)=0.33', 'PARP(86)=0.66', 'PARP(89)=1000.', 'PARP(91)=1.0', 'MSTJ(11)=3', 'MSTJ(22)=2'),
        pythiaJets = cms.vstring('MSUB(11)=1', 'MSUB(12)=1', 'MSUB(13)=1', 'MSUB(28)=1', 'MSUB(53)=1', 'MSUB(68)=1'),
        parameterSets = cms.vstring('pythiaDefault'),
        pythiaPromptPhotons = cms.vstring('MSUB(14)=1', 'MSUB(18)=1', 'MSUB(29)=1', 'MSUB(114)=1', 'MSUB(115)=1'),
        pythiaDefault = cms.vstring('MSEL=1', 'MSTU(21)=1', 'PARU(14)=1.', 'MSTP(81)=0', 'PMAS(5,1)=4.8', 'PMAS(6,1)=175.0', 'CKIN(3)=7.')
    ),
    qgpProperTimeFormation = cms.double(0.1)
)

# whatever parameters from the hydjetSourceDefault you want to modify, do it following the exampels
HydjetSource.bFixed = 0.
# replace HydjetSource.bMax          = 0.               # max impact param (fm)
# replace HydjetSource.bMin          = 0.               # min impact param (fm)
HydjetSource.cFlag = 0
# =  0 fixed impact param
# <> 0 between bmin and bmax
HydjetSource.nMultiplicity = 26000
#automatically calculated for other centralities and beams
HydjetSource.hydjetMode = 'kHydroQJets'


