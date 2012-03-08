import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(0),
    comEnergy = cms.double(8000.0),
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            TauolaPolar,
            TauolaDefaultInputCards
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    UseExternalGenerators = cms.untracked.bool(True),

    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0          ! User defined processes',
                                        'MSTJ(1)=1       ! Fragmentation/hadronization on or off'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings',
                                    'processParameters')
        ),

    jetMatching = cms.untracked.PSet(
        scheme = cms.string("Alpgen"),
        applyMatching = cms.bool(True),
        exclusive = cms.bool(True),
        etMin = cms.double(48.0),
        drMin = cms.double(0.7)
        )
)
