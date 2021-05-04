import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('D*+ -> D0 pi+ -> (K-pi+pi+pi-) pi+ at 13TeV'),
    name = cms.untracked.string('$Source: test_d0k4pi_ptdstarcut_gen_13TeV.py,v $')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('4pi_ptdstarcut_gen.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)


# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
#from Configuration.Generator.PythiaUESettings_cfi import * #for 8 TeV
from Configuration.Generator.PythiaUEZ2starSettings_cfi import * #for 13 TeV
#################
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(2),
    comEnergy = cms.double(13000.0),    
    ExternalDecays = cms.PSet(
        EvtGen = cms.untracked.PSet(
             operates_on_particles = cms.vint32( 0 ), # 0 (zero) means default list (hardcoded)
                                                      # you can put here the list of particles (PDG IDs)
                                                      # that you want decayed by EvtGen
         use_default_decay = cms.untracked.bool(False),
             decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY_NOLONGLIFE.DEC'),
              # decay_table = cms.FileInPath('GeneratorInterface/ExternalDecays/data/DECAY.DEC'),
              particle_property_file = cms.FileInPath('GeneratorInterface/ExternalDecays/data/evt.pdl'),
              user_decay_file = cms.FileInPath('Dati13/TrackEfficiency/Dstar_D0_K3pi.dec'),
              list_forced_decays = cms.vstring('MyD*+','MyD*-')
              ),
         parameterSets = cms.vstring('EvtGen')
    ),
    PythiaParameters = cms.PSet(
      #process.pythiaUESettingsBlock,
      pythiaUESettingsBlock,
      ccbarSettings= cms.vstring('MSEL=4     ! ccbar '),
      # This is a vector of ParameterSet names to be read, in this order
      parameterSets = cms.vstring('pythiaUESettings','ccbarSettings')
    )    
)

#process.D0DecayFilter = cms.EDFilter("PythiaDauFilter",
#    ParticleID = cms.untracked.int32(421),
#    ChargeConjugation = cms.untracked.bool(True),
#    MinEta = cms.untracked.double(-100.),
#    MaxEta = cms.untracked.double(100.),
#    DaughterIDs = cms.untracked.vint32(-321,211,211,-211),
#    NumberDaughters = cms.untracked.int32(4)
#)

process.DstarFilter = cms.EDFilter("PythiaMomDauFilter",
    ParticleID = cms.untracked.int32(413),
    DaughterID = cms.untracked.int32(421),
    ChargeConjugation = cms.untracked.bool(True),
    MinEta = cms.untracked.double(-100.),
    MaxEta = cms.untracked.double(100.),
    DaughterIDs = cms.untracked.vint32(421,211),
    NumberDaughters = cms.untracked.int32(2),
    MomMinPt = cms.untracked.double(3.9),
    NumberDescendants = cms.untracked.int32(4),
    DescendantsIDs = cms.untracked.vint32(-321,211,211,-211)
)

#################

#process.ProductionFilterSequence = cms.Sequence(process.generator*process.DstarFilter*process.D0DecayFilter)
process.ProductionFilterSequence = cms.Sequence(process.generator*process.DstarFilter)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
#process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
#process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.RAWSIMoutput_step)
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# End of customisation functions


	
