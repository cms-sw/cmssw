import FWCore.ParameterSet.Config as cms
process = cms.Process('VALIDATION')

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


process.generator = cms.EDFilter("Pythia8GeneratorFilter",
                                 ExternalDecays = cms.PSet(
                                   EvtGen130 = cms.untracked.PSet(
                                     operates_on_particles = cms.vint32( 0 ), # 0 (zero) means default list (hardcoded)
                                                                              # you can put here the list of particles (PDG IDs)
                                                                              # that you want decayed by EvtGen
                                     use_default_decay = cms.untracked.bool(False),
                                     decay_table = cms.string('GeneratorInterface/EvtGenInterface/data/DECAY_2010.DEC'),
                                     particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
                                     #user_decay_file = cms.vstring('GeneratorInterface/ExternalDecays/data/LambdaB_JPsiLambda_ppi.dec'),
                                     user_decay_file = cms.vstring('GeneratorInterface/EvtGenInterface/data/LambdaB_pmunu_LCSR.dec'),
                                     list_forced_decays = cms.vstring('MyLambda_b0','Myanti-Lambda_b0'),
                                     B_Mixing = cms.untracked.int32(1),
                                     #particles_to_polarize = cms.untracked.vint32(5122, -5122),
                                     #particle_polarizations = cms.untracked.vdouble(-0.4, -0.4),
                                   ),
                                  parameterSets = cms.vstring('EvtGen130'),
                                 ),
                                   maxEventsToPrint = cms.untracked.int32(0),
                                   pythiaPylistVerbosity = cms.untracked.int32(1),
                                   filterEfficiency = cms.untracked.double(1.0), #service param
                                   pythiaHepMCVerbosity = cms.untracked.bool(False),
                                   comEnergy = cms.double(7000.0),
                                   crossSection = cms.untracked.double(0.0), #service param
                                   UseExternalGenerators = cms.untracked.bool(True),
                                   PythiaParameters = cms.PSet(
                                           pythia8CommonSettingsBlock,
                                           pythia8CUEP8M1SettingsBlock,
                                           ##This would be equivalent to msel=5.
                                           processParameters = cms.vstring('HardQCD:hardbbbar = on' #equivalent to HardQCD:gg2bbbar = on','HardQCD:qqbar2bbbar = on'
                                           ##Include all other hard proceses (recommended).
                                           #processParameters = cms.vstring('HardQCD:all = on'
                                           ##This is equivalent to MSEL=1, which includes soft and hard processes.
                                           #processParameters = cms.vstring('SoftQCD:nonDiffractive = on' # previously called 'SoftQCD:minBias = on'
                                          ),
                                         parameterSets = cms.vstring('pythia8CommonSettings',
                                                                     'pythia8CUEP8M1Settings',
                                                                     'processParameters',
                                                                    )
                                    )
)



process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('TestEvtGen.root')
)

process.p = cms.Path(process.generator)
process.outpath = cms.EndPath(process.GEN)

process.schedule = cms.Schedule(process.p, process.outpath)


