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
    'MultipleInteractions:pT0Ref=2.1006',
    'MultipleInteractions:ecmPow=0.21057',
    'MultipleInteractions:expPow=1.6089',
    'BeamRemnants:reconnectRange=3.31257',
    ),
    parameterSets = cms.vstring('processParameters')
    )
                         )
