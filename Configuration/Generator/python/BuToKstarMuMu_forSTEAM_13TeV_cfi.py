# Only for Trigger Study

import FWCore.ParameterSet.Config as cms


configurationMetadata = cms.untracked.PSet(
version = cms.untracked.string('$Revision: 1.1 $'),
name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/GenProduction/python/EightTeV/PYTHIA6_BuToKstarMuMu_EtaPtFilter_TuneZ2star_8TeV_cff.py,v $'),
annotation = cms.untracked.string('Summer11: Pythia6+EvtGen generation of Bu --> K* Mu+Mu-, 8TeV, D6T tune'))


from Configuration.Generator.PythiaUEZ2starSettings_cfi import *

generator = cms.EDFilter("Pythia6GeneratorFilter",
pythiaPylistVerbosity = cms.untracked.int32(0),
pythiaHepMCVerbosity = cms.untracked.bool(False),
comEnergy = cms.double(13000.0),
crossSection = cms.untracked.double(54000000000), # Given by PYTHIA after running
filterEfficiency = cms.untracked.double(0.004), # Given by PYTHIA after running
maxEventsToPrint = cms.untracked.int32(0),

ExternalDecays = cms.PSet(
        EvtGen = cms.untracked.PSet(
operates_on_particles = cms.vint32(0),
use_default_decay = cms.untracked.bool(False),
decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY.DEC'),
particle_property_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/evt.pdl'),
user_decay_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/Bu_Kstarmumu_Kspi.dec'),
list_forced_decays = cms.vstring('MyB+','MyB-')),
        parameterSets = cms.vstring('EvtGen')),

PythiaParameters = cms.PSet(
pythiaUESettingsBlock,
bbbarSettings = cms.vstring('MSEL = 1'),
parameterSets = cms.vstring('pythiaUESettings','bbbarSettings')))


###########
# Filters #
###########
# Filter only pp events which produce a B+:
bufilter = cms.EDFilter("PythiaFilter", ParticleID = cms.untracked.int32(521))

# Filter on final state muons
mumugenfilter = cms.EDFilter("MCParticlePairFilter",
Status = cms.untracked.vint32(1, 1),
MinPt = cms.untracked.vdouble(2.8, 2.8),
MaxEta = cms.untracked.vdouble(2.3, 2.3),
MinEta = cms.untracked.vdouble(-2.3, -2.3),
ParticleCharge = cms.untracked.int32(-1),
ParticleID1 = cms.untracked.vint32(13),
ParticleID2 = cms.untracked.vint32(13))

ProductionFilterSequence = cms.Sequence(generator*bufilter*mumugenfilter)
