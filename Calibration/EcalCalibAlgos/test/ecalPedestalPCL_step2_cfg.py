import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

# my client and my Tests
process.load('DQMServices.Examples.test.DQMExample_Step2_cfi')
process.load('DQMServices.Examples.test.DQMExample_GenericClient_cfi')
process.load('DQMServices.Examples.test.DQMExample_qTester_cfi')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:OUT_step1.root"))


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')  

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:MyPedestals.db'

# A data source must always be defined. We don't need it, so here's a dummy one.
#process.source = cms.Source("EmptyIOVSource",
#    timetype = cms.string('runnumber'),
#    firstValue = cms.uint64(1),
#    lastValue = cms.uint64(1),
#    interval = cms.uint64(1)
#)

# We define the output service.
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('myPedestal_test')
    ))
)



# Path and EndPath definitions
process.load('Calibration.EcalCalibAlgos.ecalPedestalPCLHarvester_cfi')

process.myHarvesting = cms.Path(process.ECALpedestalPCLHarvester)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(
                                process.myHarvesting,
                                process.dqmsave_step
    )

process.DQMStore.verbose =  cms.untracked.int32(1)
process.DQMStore.verboseQT =  cms.untracked.int32(1)


process.dqmSaver.workflow = '/ECALPedestals/A/B'
