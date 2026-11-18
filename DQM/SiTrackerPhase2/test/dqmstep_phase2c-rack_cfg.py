##Takes as input GEN-SIM-RECO
# Auto generated configuration file
# using: 
# Revision: 1.19 

# This configuration file runs the first step of the DQM sequence.
# Usage:
# cmsRun dqmstep_phase2tk_cfg.py
# Output:
# step3_pre4_inDQM.root - input for the next step in the DQM sequence, harvestingstep_phase2tk_cfg.py

# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:phase2_realistic_T21 -s RAW2DIGI,L1Reco,RECO,RECOSIM,VALIDATION:@phase2Validation,DQM:@phase2 --datatier DQMIO -n 10 --geometry Extended2026D98 --era Phase2C11M9 --eventcontent DQM --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9

process = cms.Process('RERECO',Phase2C11M9)

# set this to true if you're using real Cosmic Rack data (not MC)
REAL_DATA = False

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D500Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
if (REAL_DATA):
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring('file:/eos/project-c/cms-tracker-integration/www/results/CosmicRackData/2026/9/Run_16_09_2026/RECO/DTH_Decoded_CRACK_Physics_Run_16th_September_2026_5_Ladders_Source_4_RECO.root',
'file:/eos/project-c/cms-tracker-integration/www/results/CosmicRackData/2026/9/Run_16_09_2026/RECO/DTH_Decoded_CRACK_Physics_Run_16th_September_2026_5_Ladders_Source_5_RECO.root',
'file:/eos/project-c/cms-tracker-integration/www/results/CosmicRackData/2026/9/Run_16_09_2026/RECO/DTH_Decoded_CRACK_Physics_Run_16th_September_2026_5_Ladders_Source_6_RECO.root',
'file:/eos/project-c/cms-tracker-integration/www/results/CosmicRackData/2026/9/Run_16_09_2026/RECO/DTH_Decoded_CRACK_Physics_Run_16th_September_2026_5_Ladders_Source_7_RECO.root'),
        secondaryFileNames = cms.untracked.vstring(),
        duplicateCheckMode = cms.untracked.string("noDuplicateCheck")
    )
else:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring('file:/eos/user/a/alaperto/CRACK/CRACK_MC_20M_45theta_14phi_r45_Zcenter.root'),
        secondaryFileNames = cms.untracked.vstring()
    )

process.options = cms.untracked.PSet(
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    fileMode = cms.untracked.string('FULLMERGE'),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False),
    TryToContinue = cms.untracked.vstring('ProductNotFound')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('step3_pre4_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.trackerGeometry.applyAlignment = False
process.mix.playback = True
for a in process.aliases: delattr(process, a)
if (REAL_DATA):
    process.RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string("")
else:
    process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.recosim_step = cms.Path(process.recosim)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

#DQM modules
process.load('DQM.SiTrackerPhase2.Phase2CRackDQMFirstStep_cff')
process.otdqm_seq = cms.Sequence(process.trackerphase2DQMSource.copy())
process.dqm_step=cms.Path(process.otdqm_seq)

if (REAL_DATA):
    process.clusterMonitorCRACK.clusterSrc = cms.InputTag("Unpacker", "", "UNPACK")
    process.dqm_step.remove(process.CRACKDigiMon)
    process.dqm_step.remove(process.Phase2OTMonitorTTCluster)
    process.dqm_step.remove(process.TTStubMonitorCRACK)
    process.dqm_step.remove(process.Phase2OTMonitorTTTrack)


# Schedule definition
process.schedule = cms.Schedule(process.dqm_step,
                                process.DQMoutput_step
)
# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions


# Customisation from command line

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
