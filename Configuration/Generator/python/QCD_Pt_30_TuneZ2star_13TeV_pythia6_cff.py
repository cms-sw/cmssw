import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *


generator = cms.EDFilter("Pythia6GeneratorFilter",
comEnergy = cms.double(13000.0),
crossSection = cms.untracked.double(6.04110e+07),
filterEfficiency = cms.untracked.double(1.0000),
maxEventsToPrint = cms.untracked.int32(0),
pythiaHepMCVerbosity = cms.untracked.bool(False),
pythiaPylistVerbosity = cms.untracked.int32(0),

PythiaParameters = cms.PSet(
pythiaUESettingsBlock,
processParameters = cms.vstring(
'MSEL=1 ! QCD hight pT processes',
'CKIN(3)=30 ! minimum pt hat for hard interactions',
),
parameterSets = cms.vstring(
'pythiaUESettings',
'processParameters',
)
)
)

ProductionFilterSequence = cms.Sequence(generator)
