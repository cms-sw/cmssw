from __future__ import print_function
#########################
#
# Configuration file for PileUp events
# production in tracker only
#
# Author: S.Viret (viret@in2p3.fr)
# Date        : 16/02/2017
#
# Script tested with release CMSSW_10_0_0_pre1
#
#########################
#
# Here you choose if you want flat (True) or tilted (False) geometry
#

flat=False

###################

import FWCore.ParameterSet.Config as cms

process = cms.Process('STUBS')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

if flat:
	print('You choose the flat geometry')
	process.load('L1Trigger.TrackTrigger.TkOnlyFlatGeom_cff') # Special config file for TkOnly geometry
	process.TTStubAlgorithm_official_Phase2TrackerDigi_.zMatchingPS = cms.bool(False) # Tilted is the new default
	process.TTStubAlgorithm_official_Phase2TrackerDigi_.EndcapCutSet = cms.VPSet(
		cms.PSet( EndcapCut = cms.vdouble( 0 ) ),
		cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 2, 3.5, 2, 3.5, 5.5, 6, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7, 7) ),
		cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 1.5, 3, 2, 3, 5, 6, 6.5, 6.5, 6.5, 5, 6.5, 6.5, 7, 7) ),
		cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 0.5, 0.5, 1, 1.5, 3, 4.5, 6, 6.5, 6.5, 7, 7, 7, 7, 7) ),
		cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 0.5, 0.5, 0.5, 1.5, 2., 3.5, 5., 6.5, 6.5, 6.5, 6, 7, 7, 7) ),
		cms.PSet( EndcapCut = cms.vdouble( 0, 0.5, 0.5, 0.5, 0.5, 1., 1.5, 2.5, 4., 5, 7, 5.5, 7, 7, 7, 7) ),
		)
else:
	print('You choose the tilted geometry')
	process.load('L1Trigger.TrackTrigger.TkOnlyTiltedGeom_cff') # Special config file for TkOnly geometry

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# Input source
process.source = cms.Source("EmptySource")

process.mix.minBunch = cms.int32(-12)
process.mix.bunchspace=cms.int32(25)
process.mix.input.nbPileupEvents.averageNumber = cms.double(20.0)  # The average number of pileup events you want  
process.mix.input.fileNames     = cms.untracked.vstring('file:MBias_100_TkOnly.root') # The file where to pick them up

# Additional output definition

process.genstepfilter.triggerConditions=cms.vstring("generation_step")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.RandomNumberGeneratorService.generator.initialSeed      = 20
process.RandomNumberGeneratorService.VtxSmeared.initialSeed     = 2
process.RandomNumberGeneratorService.g4SimHits.initialSeed      = 178
process.RandomNumberGeneratorService.mix.initialSeed            = 210


# Generate particle gun events
process.generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MaxPt = cms.double(200.0),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359),
        MinPt = cms.double(0.9),
        ParticleID = cms.vint32(-13, -13)
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('Four mu pt 1 to 200')
)


# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('PU_20_sample_TkOnly.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

process.RAWSIMoutput.outputCommands.append('keep  *_*_*_*')
process.RAWSIMoutput.outputCommands.append('drop  *_mix_*_STUBS')
process.RAWSIMoutput.outputCommands.append('drop  PCaloHits_*_*_*')
process.RAWSIMoutput.outputCommands.append('drop  *_ak*_*_*')
process.RAWSIMoutput.outputCommands.append('drop  *_simSi*_*_*')
process.RAWSIMoutput.outputCommands.append('keep  *_*_MergedTrackTruth_*')
process.RAWSIMoutput.outputCommands.append('keep  *_mix_Tracker_*')


# Path and EndPath definitions
process.generation_step         = cms.Path(process.pgen)
process.simulation_step         = cms.Path(process.psim)
process.genfiltersummary_step   = cms.EndPath(process.genFilterSummary)
process.digitisationTkOnly_step = cms.Path(process.pdigi_valid)
process.L1TrackTrigger_step     = cms.Path(process.TrackTriggerClustersStubs)
process.L1TTAssociator_step     = cms.Path(process.TrackTriggerAssociatorClustersStubs)
process.endjob_step             = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step       = cms.EndPath(process.RAWSIMoutput)


process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisationTkOnly_step,process.L1TrackTrigger_step,process.L1TTAssociator_step,process.endjob_step,process.RAWSIMoutput_step)

# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq


# Automatic addition of the customisation function
from L1Trigger.TrackTrigger.TkOnlyDigi_cff import TkOnlyDigi
process = TkOnlyDigi(process)
# End of customisation functions



