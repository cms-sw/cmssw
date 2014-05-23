import FWCore.ParameterSet.Config as cms

from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *
generator = cms.EDFilter("Pythia8HadronizerFilter",
                         ExternalDecays = cms.PSet(
    Tauola = cms.untracked.PSet(
    TauolaPolar,
    TauolaDefaultInputCards
    ),
    parameterSets = cms.vstring('Tauola')
    ),
                         UseExternalGenerators = cms.untracked.bool(True),
                         
                         maxEventsToPrint = cms.untracked.int32(1),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(13000.),
                         PythiaParameters = cms.PSet(
    processParameters = cms.vstring(
    'Main:timesAllowErrors = 10000',
    'ParticleDecays:tauMax = 10',
    'Tune:ee 3',
    'Tune:pp 5',
    'PDF:useLHAPDF=on',
    'PDF:LHAPDFset=HERAPDF1.5LO_EIG.LHgrid',
    'MultipleInteractions:pT0Ref=2.000072e+00',
    'MultipleInteractions:ecmPow=2.498802e-01',
    'MultipleInteractions:expPow=1.690506e+00',
    'BeamRemnants:reconnectRange=6.096364e+00',
    ),
    parameterSets = cms.vstring('processParameters')
    )
                         )
