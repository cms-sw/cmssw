# Auto generated configuration file
# using: 
# Revision: 1.381.2.27 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step.py --step=DIGI,L1,DIGI2RAW,HLT:my_8e33_v4 -n 45 --eventcontent HLTDEBUG --conditions auto:startup_8E33v2 --mc --fileout output.root --no_exec --python_filename hlt_my_8e33_13TeV_v1.py --datamix NODATAMIXER --datatier GEN-SIM-RAW --filein= /store/mc/Summer13dr53X/TTbar_TuneZ2star_13TeV-pythia6-tauola/GEN-SIM-RAW/PU25bx25_START53_V19D-v1/20000/0068A47D-17E3-E211-8383-003048D4604C.root --processName HLTX
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

#process.load('HLTrigger.Configuration.HLT_my_8e33_v4_cff')
process.load('HLTrigger.Configuration.HLT_8e33_2pt1v4_DAVect_Iter_ALL_DAVectInPFlowNoPU_AC7')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
'/store/mc/Summer13dr53X/TTbar_TuneZ2star_13TeV-pythia6-tauola/GEN-SIM-RAW/PU25bx25_START53_V19D-v1/20000/50B096FD-A1DF-E211-BEA7-001E67398E49.root'
)
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.27 $'),
    annotation = cms.untracked.string('step.py nevts:45'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition
from Configuration.EventContent.EventContent_cff import *
process.HLTDEBUGEventContent.outputCommands.extend(   cms.untracked.vstring(  'keep *_genParticles_*_*',      'keep *_genParticlesForJets_*_*') )

process.HLTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.HLTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('output.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RAW')
    )
)

# Additional output definition

# Other statements
from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup_8E33v2', '')
# add following line
process.GlobalTag.globaltag = 'START53_V27::All' 


print "OK1"

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.HLTDEBUGoutput_step = cms.EndPath(process.HLTDEBUGoutput)

print "OK2"


# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.HLTDEBUGoutput_step])

print "OK3"


# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC 

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions
