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
         
                            #fileNames = cms.untracked.vstring('/store/data/Run2016D/TestEnablesEcalHcal/RAW/v2/000/276/315/00000/92AB6184-FD41-E611-A550-02163E01464C.root'),
                            #fileNames = cms.untracked.vstring('/store/data/Run2016D/TestEnablesEcalHcal/RAW/v2/000/276/318/00000/8487C425-1242-E611-81E6-02163E0146AF.root'),
                            fileNames = cms.untracked.vstring(
        #'file://store_data_Run2016D_TestEnablesEcalHcal_RAW_v2_000_276_315_00000_92AB6184-FD41-E611-A550-02163E01464C.root',
        #'file://store_data_Run2016D_TestEnablesEcalHcal_RAW_v2_000_276_318_00000_8487C425-1242-E611-81E6-02163E0146AF.root',
        'file:///afs/cern.ch/work/a/argiro/ecalpedPCL/data/store_data_Run2016D_TestEnablesEcalHcal_RAW_v2_000_276_318_00000_5A25B228-1242-E611-BA75-02163E0120B0.root',
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
#process.ecalpedstalPCL = cms.EDAnalyzer('ECALpedestalPCLworker',
#                                        BarrelDigis=cms.InputTag('ecalDigis','ebDigis'),
#                                        EndcapDigis=cms.InputTag('ecalDigis','eeDigis'))

process.load('Calibration.EcalCalibAlgos.ecalPedestalPCLworker_cfi')

process.raw2digi_step = cms.Path(process.ecalDigis*process.ecalpedestalPCL)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",                                      
                                     fileName = cms.untracked.string("OUT_step1.root"))

process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.DQMoutput_step)



process.ecalDigis.InputLabel = cms.InputTag('hltEcalCalibrationRaw')
