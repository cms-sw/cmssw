# Auto generated configuration file
# using: 
# Revision: 1.20 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: TenElePt_0_20_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO --conditions auto:upgradePLS3 -n -1 --eventcontent FEVTDEBUG -s DIGI,L1,DIGI2RAW,L1TrackTrigger,RECO:pixeltrackerlocalreco --datatier GEN-SIM-DIGI-RAW --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_phase2_BE5D --geometry ExtendedPhase2TkBE5D --magField 38T_PostLS1 --filein file:TenElePt_0_20_ExtendedPhase2TkBE5D_10000_GEN_SIM.root --fileout file:TenElePt_0_20_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
#process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')

process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:TenElePt_0_20_ExtendedPhase2TkBE5D_10000_GEN_SIM.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('TenElePt_0_20_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO nevts:-1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('file:TenElePt_0_20_ExtendedPhase2TkBE5D_10000_DIGI_L1_DIGI2RAW_L1TT_RECO.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.L1TrackTrigger_step = cms.Path(process.TrackTriggerClustersStubs)
process.L1TTAssociator_step = cms.Path(process.TrackTriggerAssociatorClustersStubs)

process.reconstruction_step = cms.Path(process.pixeltrackerlocalreco)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

process.eca = cms.EDAnalyzer("EventContentAnalyzer")
process.eca_step = cms.Path(process.eca)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,
process.L1TrackTrigger_step,process.L1TTAssociator_step,
process.reconstruction_step,process.endjob_step,process.FEVTDEBUGoutput_step)#,process.eca_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_phase2_BE5D 

#call to customisation function cust_phase2_BE5D imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_phase2_BE5D(process)

# End of customisation functions
