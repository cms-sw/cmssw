import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *

generator = cms.EDFilter("ExhumeGeneratorFilter",
    ExhumeParameters = cms.PSet(
        AlphaEw = cms.double(0.0072974),
        B = cms.double(4.0),
        MuonMass = cms.double(0.1057),
        BottomMass = cms.double(4.65),
        CharmMass = cms.double(1.28),
        StrangeMass = cms.double(0.095),
        TauMass = cms.double(1.78),
        TopMass = cms.double(172.8),
        WMass = cms.double(80.38),
        ZMass = cms.double(91.187),
        HiggsMass = cms.double(125.1),
        HiggsVev = cms.double(246.0),
        LambdaQCD = cms.double(80.0),
        MinQt2 = cms.double(0.64),
        PDF = cms.double(11000),
        Rg = cms.double(1.2),
        Survive = cms.double(0.03)
    ),
    ExhumeProcess = cms.PSet(
        MassRangeHigh = cms.double(2000.0),
        MassRangeLow = cms.double(300.0),
        ProcessType = cms.string('GG'),
        ThetaMin = cms.double(0.3)
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    ),
    comEnergy = cms.double(14000.0),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1)
)

# Production Info
configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('GluGluTo2Jets 300 < Mx < 2000 7TeV ExHume'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

ProductionFilterSequence = cms.Sequence(generator)

