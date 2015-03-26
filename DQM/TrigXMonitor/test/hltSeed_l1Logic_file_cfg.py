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
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_R_35X_V1::All'

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500))

#process.source = cms.Source ("PoolSource", fileNames = cms.untracked.vstring(
#'/store/data/BeamCommissioning09/MinimumBias/RECO/v2//000/124/120/04092AB7-75E8-DE11-958F-000423D98750.root' ));

# for RAW data, run first the RAWTODIGI 
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

if dataType == 'RAW' : 

    if runNumber == 123596 :
        dataset = '/MinimumBias/BeamCommissioning09-v1/RAW'

        readFiles.extend( [
#'/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/123/596/8E21B4C8-74E2-DE11-ABAA-000423D999CA.root' 
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/000F1430-15E9-DE11-ACB9-000423D94E70.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/0049E59B-11E9-DE11-801C-001617C3B778.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/0055F768-08E9-DE11-AF95-001D09F28EA3.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/00D1AFBC-07E9-DE11-8288-001D09F231B0.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/084BFE68-08E9-DE11-A76F-001D09F24600.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/08772497-0AE9-DE11-80BF-0030486730C6.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/0C0D9BEE-10E9-DE11-AE0A-001D09F2B30B.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/0C6DF08F-0AE9-DE11-A36E-001D09F295FB.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/0CE8FB95-0AE9-DE11-983F-003048D2C020.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/1245F2C9-0EE9-DE11-8A86-001D09F251FE.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/16667F5A-12E9-DE11-896D-001D09F24EE3.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/183CC241-0BE9-DE11-BF12-001D09F2514F.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/1CFA4E18-0EE9-DE11-98C4-001D09F29114.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/1E5BB330-10E9-DE11-A5EE-001D09F2527B.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/20AE0783-0FE9-DE11-A1AA-001D09F23D1D.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/20B1AF1A-0EE9-DE11-B4FE-001D09F24F65.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/249287B4-0CE9-DE11-AD7D-000423D94524.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/268335C8-0CE9-DE11-A936-000423D6AF24.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/26A31C1A-0EE9-DE11-A4E6-001D09F23174.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/28458F63-0DE9-DE11-AC5F-0030487C6090.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/28738B10-13E9-DE11-BF33-003048D374F2.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/2A6513CB-13E9-DE11-ABE7-001D09F290CE.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/2C22885C-12E9-DE11-A408-001D09F23A20.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/34FA1F5D-0DE9-DE11-BA7E-001D09F26C5C.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/3673228D-0AE9-DE11-81B7-0019B9F7312C.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/3876A812-13E9-DE11-8EDA-001D09F232B9.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/3C315920-09E9-DE11-9077-001D09F244DE.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/4467618D-0AE9-DE11-900E-003048D375AA.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/4A7F7BAF-0AE9-DE11-B5FD-001617E30D12.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/566FC734-15E9-DE11-B36B-000423D94908.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/5AFB1D30-10E9-DE11-A24C-001D09F29524.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/6032A4B6-0CE9-DE11-9732-0030487A18F2.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/62B9F17D-14E9-DE11-BD4C-000423D99F1E.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/62EAF0F8-09E9-DE11-9555-001D09F28D4A.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/66B309F3-0BE9-DE11-9FFA-003048D37538.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/6C6F928E-25E9-DE11-B9EF-003048D37538.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/6E2E4A8B-0AE9-DE11-AF65-001D09F290CE.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/6E66A9D1-09E9-DE11-B252-001D09F250AF.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/124/230/6E8C8318-0EE9-DE11-AE64-001D09F25393.root'
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
#process.MessageLogger = cms.Service("MessageLogger",
#    detailedInfo= cms.untracked.PSet(
#      threshold = cms.untracked.string('DEBUG'),
#      DEBUG = cms.untracked.PSet(
#            limit = cms.untracked.int32(10000)
#      )
#    ),
#    critical = cms.untracked.PSet(
#        threshold = cms.untracked.string('ERROR')
#    ),
##debugModules = cms.untracked.vstring('hltResults','hltFourVectorClient'),
#    debugModules = cms.untracked.vstring('hltSeedL1Logic'),
##debugModules = cms.untracked.vstring('*'),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('WARNING'),
#        WARNING = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        noLineBreaks = cms.untracked.bool(True)
#    ),
#    destinations = cms.untracked.vstring('detailedInfo', 
#        'critical', 
#        'cout')
##    destinations = cms.untracked.vstring( 'critical', 'cout')
#)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.load("DQM.TrigXMonitor.HLTSeedL1LogicScalers_cfi")
#process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")

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
#process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
#process.pHLT = cms.Path(process.RawToDigi+process.l1GtTrigReport+process.hltSeedL1Logic)
#process.pHLT = cms.Path(process.RawToDigi+process.hltSeedL1Logic)
    process.pHLT = cms.Path(process.gtDigis+process.hltSeedL1Logic)
    
else :        
    # path to be run for RECO
    #process.pHLT = cms.Path(process.l1GtTrigReport+process.hltSeedL1Logic)
    process.pHLT = cms.Path(process.hltSeedL1Logic)


process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/StreamExpress/BeamCommissioning09-v8/FVMB'

open('dump.py', 'w').write(process.dumpPython())
