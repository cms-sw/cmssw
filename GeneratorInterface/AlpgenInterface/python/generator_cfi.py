import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *

generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(14000.0),

    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0          ! User defined processes',
                                        'MSTJ(1)=1       ! Fragmentation/hadronization on or off',
                                        'MSTJ(11)=3      ! Choice of the fragmentation function'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings',
                                    'processParameters')
        ),

    jetMatching = cms.untracked.PSet(
        scheme = cms.string("Alpgen"),
        applyMatching = cms.bool(True),
        exclusive = cms.bool(True),
        etMin = cms.double(25.),
        drMin = cms.double(0.7)
        )
)
