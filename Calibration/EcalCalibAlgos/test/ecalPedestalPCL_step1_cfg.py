# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: data -s RAW2DIGI -n 500 --filein=file:/build/argiro/data/MinBias-Run2011B-RAW.root --data --conditions auto --scenario pp --process RERECO
import FWCore.ParameterSet.Config as cms

process = cms.Process('RERECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
     '/store/data/Run2017B/TestEnablesEcalHcal/RAW/v1/000/299/149/00000/24768806-E869-E711-A7F2-02163E019BBE.root',
     '/store/data/Run2017B/TestEnablesEcalHcal/RAW/v1/000/299/149/00000/26D7BF02-E869-E711-842E-02163E01A1CE.root',
     '/store/data/Run2017B/TestEnablesEcalHcal/RAW/v1/000/299/149/00000/2AED4B13-EB69-E711-A167-02163E01A5B3.root',         
                                  ), 
                            secondaryFileNames = cms.untracked.vstring(),
                            #duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
 )

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('data nevts:500'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')



# Path and EndPath definitions

process.load('Calibration.EcalCalibAlgos.ecalPedestalPCLworker_cfi')


process.tcdsDigis = cms.EDProducer('TcdsRawToDigi', InputLabel=cms.InputTag('hltEcalCalibrationRaw'))
process.raw2digi_step = cms.Path(process.tcdsDigis*process.ecalDigis*process.ecalpedestalPCL)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",                                      
                                     fileName = cms.untracked.string("OUT_step1.root"))

process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.DQMoutput_step)


process.ecalDigis.InputLabel = cms.InputTag('hltEcalCalibrationRaw')

