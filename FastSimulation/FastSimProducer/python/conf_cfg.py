import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process("DEMO",eras.Run2_2016,eras.fastSim)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# load particle data table
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
# load geometry
process.load('FastSimulation.Configuration.Geometries_MC_cff')
# load magnetic field
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load("Configuration.StandardSequences.MagneticField_0T_cff") 
#load and set conditions (required by geometry and magnetic field)
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')          

# read generator event from file
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:gen_muGun.root'),
)

# configure random number generator for simhit production
process.load('Configuration.StandardSequences.Services_cff')
process.RandomNumberGeneratorService = cms.Service(
    "RandomNumberGeneratorService",
    fastSimProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(234567),
        engineName = cms.untracked.string('TRandom3')
        ),
    fastTrackerRecHits = cms.PSet(
         initialSeed = cms.untracked.uint32(24680),
         engineName = cms.untracked.string('TRandom3')
     ),
    )

# output
process.load('Configuration.EventContent.EventContent_cff')
process.load('FastSimulation.Configuration.Reconstruction_BefMix_cff')

# load simhit producer
process.load("FastSimulation.FastSimProducer.fastSimProducer_cff")

# Output definition
process.FEVTDEBUGHLTEventContent.outputCommands.append(
        'keep *_fastSimProducer_*_*',
    )

#process.FEVTDEBUGHLTEventContent.outputCommands = cms.untracked.vstring(
#        'keep *',
#    )

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(10485760),
    fileName = cms.untracked.string('res_muGun.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('dqm_res_muGun.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# use new TrackerSimHitProducer
process.fastTrackerRecHits.simHits = cms.InputTag("fastSimProducer","TrackerHits")
process.fastMatchedTrackerRecHits.simHits = cms.InputTag("fastSimProducer","TrackerHits")
process.fastMatchedTrackerRecHitCombinations.simHits = cms.InputTag("fastSimProducer","TrackerHits")

# define a path to run
process.simulation_step = cms.Path(process.fastSimProducer)

process.reconstruction_befmix_step = cms.Path(process.reconstruction_befmix)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
#process.schedule = cms.Schedule(process.simulation_step,process.FEVTDEBUGHLToutput_step,process.DQMoutput_step)
process.schedule = cms.Schedule(process.simulation_step,process.reconstruction_befmix_step,process.FEVTDEBUGHLToutput_step,process.DQMoutput_step)

# debugging options
# debug messages will only be printed for packages compiled with following command
# USER_CXXFLAGS="-g -D=EDM_ML_DEBUG" scram b -v # for bash
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        ),
    debugModules = cms.untracked.vstring('fastSimProducer')
    )

