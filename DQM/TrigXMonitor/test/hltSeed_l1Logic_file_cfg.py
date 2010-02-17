import FWCore.ParameterSet.Config as cms
import sys

dataType = 'RAW'
runNumber = 123596

process = cms.Process("DQM")

#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_R_35X_V1::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))

#process.source = cms.Source ("PoolSource", fileNames = cms.untracked.vstring(
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2//000/124/120/04092AB7-75E8-DE11-958F-000423D98750.root' ));

# for RAW data, run first the RAWTODIGI 
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

if dataType == 'RAW' : 

    if runNumber == 123596 :
        dataset = '/Cosmics/BeamCommissioning09-v1/RAW'

        readFiles.extend( [
            '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/123/596/8E21B4C8-74E2-DE11-ABAA-000423D999CA.root' 
            ] );

        secFiles.extend([
            ])


    elif runNumber == 116035 :
        dataset = '/Cosmics/Commissioning09-v3/RAW'
        print '   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag 

        readFiles.extend( [                        
            '/store/data/Commissioning09/Cosmics/RAW/v3/000/116/035/34A8317D-76AF-DE11-91DB-000423D98DC4.root'
            ]);                                                                                               

        secFiles.extend([
            ])
    
    elif runNumber == 121560 :
        dataset = '/Cosmics/Commissioning09-v3/RAW'
        print '   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag 

        readFiles.extend( [                        
            '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/560/DC089E4B-5ED4-DE11-A179-000423D98FBC.root'
            ]);                                                                                               

        secFiles.extend([
            ])

    else :
        print 'Error: run ', runNumber, ' not defined.'    
        sys.exit()


elif dataType == 'FileStream' : 
    # data dat
    readFiles.extend( [
            'file:/lookarea_SM/MWGR_29.00105760.0001.A.storageManager.00.0000.dat'
    
        ] );

elif dataType == 'RECO' : 

    if runNumber == 123596 :
        dataset = '/Cosmics/BeamCommissioning09-v1/RECO'

        readFiles.extend( [
            '/store/data/BeamCommissioning09/Cosmics/RECO/v2/000/123/596/FC5C3B0F-8AE2-DE11-A905-003048D37456.root' 
            ] );

        secFiles.extend([
            ])

#
process.MessageLogger = cms.Service("MessageLogger",
    detailedInfo= cms.untracked.PSet(
      threshold = cms.untracked.string('DEBUG'),
      DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(10000)
      )
    ),
    critical = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
#debugModules = cms.untracked.vstring('hltResults','hltFourVectorClient'),
    debugModules = cms.untracked.vstring('hltSeedL1Logic'),
#debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('detailedInfo', 
        'critical', 
        'cout')
#    destinations = cms.untracked.vstring( 'critical', 'cout')
)

process.load("DQM.TrigXMonitor.HLTSeedL1LogicScalers_cfi")
process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")

# boolean flag to select the input record
#process.l1GtTrigReport.UseL1GlobalTriggerRecord = True

# input tag for the GT record requested: 
#   GT emulator:    gtDigis (DAQ record)
#   GT unpacker:    gtDigis (DAQ record)
#   GT lite record: l1GtRecord 
#process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"

#process.l1GtTrigReport.PrintVerbosity = 2

# print output: 0 = std::cout; 1 = LogTrace; 2 = LogVerbatim; 3 = LogInfo
#process.l1GtTrigReport.PrintOutput = 1


# for RAW data, run first the RAWTODIGI 
if dataType == 'RAW' :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
    process.pHLT = cms.Path(process.RawToDigi+process.l1GtTrigReport+process.hltSeedL1Logic)
    
else :        
    # path to be run for RECO
    process.pHLT = cms.Path(process.l1GtTrigReport+process.hltSeedL1Logic)


process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/StreamExpress/BeamCommissioning09-v8/FVMB'


