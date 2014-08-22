# Auto generated configuration file
# using: 
# Revision: 1.381.2.6 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step.py --step=DIGI,L1,DIGI2RAW,HLT:orcoff:/cdaq/physics/Run2012/7e33/v4.1/HLT/V1 --datamix NODATAMIXER --filein=/store/mc/Summer13dr53X/TTbar_TuneZ2star_13TeV-pythia6-tauola/GEN-SIM-RAW/PU25bx25_START53_V19D-v1/20000/0068A47D-17E3-E211-8383-003048D4604C.root --fileout=AODSIM_STARTUP.root --mc --no_exec --number=10 --eventcontent=FEVTDEBUGHLT --conditions=auto:startup --python_filename=test_7e33v2.py --number=100 --processName=HLTX
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLTX')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
#    fileNames = cms.untracked.vstring('/store/mc/Summer13dr53X/TTbar_TuneZ2star_13TeV-pythia6-tauola/GEN-SIM-RAW/PU25bx25_START53_V19D-v1/20000/0068A47D-17E3-E211-8383-003048D4604C.root')
    fileNames = cms.untracked.vstring(
'file:/store/skim/201624/triggerSkim_HLTPhysics_run201624_83_176.root'
)
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.6 $'),
    annotation = cms.untracked.string('step.py nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string('AODSIM_STARTUP.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
import HLTrigger.Configuration.Utilities
process.loadHltConfiguration("orcoff:/cdaq/physics/Run2012/7e33/v4.1/HLT/V1",type='GRun')
from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

process.GlobalTag.globaltag = 'START53_V9::All'

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions
