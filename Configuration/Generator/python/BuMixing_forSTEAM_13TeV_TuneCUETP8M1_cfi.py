##!!!For BPH Trigger study only
import FWCore.ParameterSet.Config as cms
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *
generator = cms.EDFilter("Pythia8GeneratorFilter",
pythiaPylistVerbosity = cms.untracked.int32(0),
pythiaHepMCVerbosity = cms.untracked.bool(False),
comEnergy = cms.double(13000.0),
maxEventsToPrint = cms.untracked.int32(0),
ExternalDecays = cms.PSet(
EvtGen130 = cms.untracked.PSet(
decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2010.DEC'),
particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/Bu_Mixing.dec'),
list_forced_decays = cms.vstring('MyB+',
'MyB-'),
operates_on_particles = cms.vint32()
),
parameterSets = cms.vstring('EvtGen130')
),
PythiaParameters = cms.PSet(
pythia8CommonSettingsBlock,
pythia8CUEP8M1SettingsBlock,
processParameters = cms.vstring(
'HardQCD:all = on'
),
parameterSets = cms.vstring('pythia8CommonSettings',
'pythia8CUEP8M1Settings',
'processParameters',
)
)
)
generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)
configurationMetadata = cms.untracked.PSet(
version = cms.untracked.string('$Revision: 1.1 $'),
name = cms.untracked.string('$Source: Configuration/Generator/python/BuToKstarMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi.py $'),
annotation = cms.untracked.string('Summer14: Pythia8+EvtGen130 generation of Bu --> K* Mu+Mu-, 13TeV, Tune CUETP8M1')
)
###########
# Filters #
###########
# Filter only pp events which produce a B+:
bufilter = cms.EDFilter("PythiaFilter", ParticleID = cms.untracked.int32(521))
ProductionFilterSequence = cms.Sequence(generator*bufilter)
