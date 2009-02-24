import FWCore.ParameterSet.Config as cms
from Configuration.Generator.PythiaUESettings_cfi import *

generator = cms.EDProducer("LHEProducer",
    eventsToPrint = cms.untracked.uint32(0),

    hadronisation = cms.PSet(
        pythiaUESettingsBlock,
        generator = cms.string('Pythia6'),
        maxEventsToPrint = cms.untracked.int32(0),
        pythiaPylistVerbosity = cms.untracked.int32(0),

        parameterSets = cms.vstring(
            'pythiaUESettings',
            'pythiaAlpgen'
            ),
        
        pythiaAlpgen = cms.vstring(
            'MSTJ(11)=3 ! Choice of the fragmentation function',
            'MSEL=0     ! User defined processes/Full user control'
            )
        ),

       jetMatching = cms.untracked.PSet(
           scheme = cms.string("Alpgen"),
           applyMatching = cms.bool(True),
           exclusive = cms.bool(True),
           etMin = cms.double(25.),
           drMin = cms.double(0.7)
           )
)
