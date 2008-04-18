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
    PythiaParameters = cms.PSet(
        orcaSettings = cms.vstring('MSEL=0', 'PARP(67)=1.', 'PARP(82)=1.9', 'PARP(85)=0.33', 'PARP(86)=0.66', 'PARP(89)=1000.', 'PARP(91)=1.0', 'MSTJ(11)=3', 'MSTJ(22)=2'),
        pythiaJets = cms.vstring('MSUB(11)=1', 'MSUB(12)=1', 'MSUB(13)=1', 'MSUB(28)=1', 'MSUB(53)=1', 'MSUB(68)=1'),
        parameterSets = cms.vstring('pythiaDefault'),
        pythiaPromptPhotons = cms.vstring('MSUB(14)=1', 'MSUB(18)=1', 'MSUB(29)=1', 'MSUB(114)=1', 'MSUB(115)=1'),
        pythiaDefault = cms.vstring('MSEL=1', 'MSTU(21)=1', 'PARU(14)=1.', 'MSTP(81)=0', 'PMAS(5,1)=4.8', 'PMAS(6,1)=175.0', 'CKIN(3)=7.')
    ),
    qgpProperTimeFormation = cms.double(0.1)
)

HydjetSource.bFixed = 0.
HydjetSource.cFlag = 0
HydjetSource.nMultiplicity = 26000
HydjetSource.hydjetMode = 'kHydroQJets'


