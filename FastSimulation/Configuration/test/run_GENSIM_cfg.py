import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('analysis')
options.outputFile = "GENSIM.root"


options.parseArguments()



# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('FastSimulation.Configuration.Geometries_MC_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedNominalCollision2015_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('FastSimulation.Configuration.FamosSequences_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("EmptySource")

# Other statements
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# configure generator for ttbar
process.load("Configuration.Generator.TTbar_13TeV_TuneCUETP8M1_cfi")

# store everything
process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string(options.outputFile),
    outputCommands = cms.untracked.vstring( ('keep *'))
    )

# Path and EndPath definitions
process.generation_step = cms.Path(process.generator)
process.simulation_step = cms.Path(process.simulationSequence)

process.path = cms.Path(process.generator*process.simulationSequence)
process.endpath = cms.EndPath(process.output)

# run2 customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 
process = customisePostLS1(process)

