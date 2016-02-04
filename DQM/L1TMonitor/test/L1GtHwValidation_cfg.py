import FWCore.ParameterSet.Config as cms
import sys

#
# cfg file to run L1GtHwValidation 
#
# V M Ghete 2009-10-09

###################### user choices ######################
# choose (pre)release
latestRelease = 'CMSSW_3_2_6'
useRelease = latestRelease

# choose the type of sample used (True for RelVal, False for data)
#useRelValSample = True 
useRelValSample=False 

if useRelValSample == True :
    
    if useRelease == latestRelease :
         #useGlobalTag = 'MC_31X_V8'
         useGlobalTag = 'STARTUP31X_V7'

    # RelVals 
    useSample = 'RelValQCD_Pt_80_120'
    #useSample = 'RelValTTbar'
    #useSample = 'RelValZTT'
 
    # data type
    dataType = 'RAW'
   
else :
    # global tag
    
    if useRelease == latestRelease :
        #useGlobalTag = 'CRAFT0831X_V1'
        useGlobalTag = 'GR09_P_V6'
    
        # data type
        dataType = 'RAW'
        #dataType = 'StreamFile'

# change to True to use local files
#     the type of file should match the choice of useRelValSample and useGlobalTag
useLocalFiles = False 

###################### end user choices ###################


process = cms.Process("DQM")

#  DQM SERVICES
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#  DQM SOURCES
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#
# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'



process.load("DQM/L1TMonitor/L1TEmulatorMonitor_cff")


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(10)
)

# for RAW data, run first the RAWTODIGI 
if dataType == 'StreamFile' :
    readFiles = cms.untracked.vstring()
    process.source = cms.Source("NewEventStreamFileReader", fileNames=readFiles)
else :        
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if useRelValSample == True :
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :
        
        if (useRelease == 'CMSSW_3_2_2') and (useSample == 'RelValQCD_Pt_80_120') :

            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/C022431D-4178-DE11-8B2E-001731AF6A89.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/68B8CFCD-7578-DE11-B953-001731AF698F.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/10B7B30E-4178-DE11-BB59-00304867929E.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/10B567B3-3F78-DE11-95DA-00304866C398.root' 
                ] );

        elif (useRelease == 'CMSSW_3_2_2') and (useSample == 'RelValTTbar') :
            
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/DC42776C-4078-DE11-9C5C-0018F3D0961E.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/7C9AB0C6-4178-DE11-A33C-003048D3FC94.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/5AFDDA6B-4078-DE11-9E9C-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/4AE1FF1C-4178-DE11-8D06-001A92971BA0.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/08950864-7578-DE11-8AB7-001731A28F19.root'
                ] );

        elif (useRelease == 'CMSSW_3_2_6') and (useSample == 'RelValQCD_Pt_80_120') :
            
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_6-MC_31X_V8-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0013/A0B4E4F5-FA9A-DE11-A2D1-001617C3B5F4.root',
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0013/986AB408-489A-DE11-A502-003048D2BF1C.root',
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0013/7052BB5E-449A-DE11-A97F-001D09F248F8.root',
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0013/5AED20CA-4D9A-DE11-A2E6-000423D6B48C.root',
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V8-v1/0013/465A7774-499A-DE11-B1AB-001D09F24F65.root'
                ] );



        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        if (useRelease == 'CMSSW_3_2_2') and (useSample == 'RelValQCD_Pt_80_120') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/A035735C-3E78-DE11-A21D-001731AF678D.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/704BF4FE-3E78-DE11-A352-003048678B0A.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/6C96FAE5-3C78-DE11-AACB-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/40911F0A-7578-DE11-9619-001A92971B88.root'
                ] );


        elif (useRelease == 'CMSSW_3_2_2') and (useSample == 'RelValTTbar') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_2_2-STARTUP31X_V2-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/D2ED5A8E-4478-DE11-BA8A-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/9ABEFAC9-7578-DE11-AA25-0018F3D095EC.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/70B9689B-3D78-DE11-B90F-001731AF6A89.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/5C711444-4578-DE11-84B5-001731A28FC9.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/0A5F22EE-3C78-DE11-8EE1-0018F3D096FE.root'            
                ] );

        elif (useRelease == 'CMSSW_3_2_2') and (useSample == 'RelValZTT') :
            
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValZMM/CMSSW_3_2_2-STARTUP31X_V2-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValZMM/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/EEA039CA-4178-DE11-B212-0018F3D09654.root',
                '/store/relval/CMSSW_3_2_2/RelValZMM/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/760DF46E-7578-DE11-A581-001A92810ADE.root',
                '/store/relval/CMSSW_3_2_2/RelValZMM/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/089417DD-4378-DE11-8AC1-001A92971ADC.root'            
                ] );

        elif (useRelease == 'CMSSW_3_2_6') and (useSample == 'RelValQCD_Pt_80_120') :
            
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_6-MC_31X_V8-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/CC073576-D89A-DE11-9B3B-001617C3B76A.root',
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/C68A4F73-DA9A-DE11-B3A9-0030487A18A4.root',
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/8A3D7D31-DC9A-DE11-BC45-0030486780B8.root',
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/7E528C8D-E09A-DE11-9956-001D09F24EE3.root',
                '/store/relval/CMSSW_3_2_6/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V7-v1/0013/5E43FD4D-DE9A-DE11-9A24-001D09F23944.root'
                ] );


        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

                
        secFiles.extend([
            ])
        
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    
        sys.exit()

else :
    if dataType == 'RAW' : 

        # data POOL
        dataset = cms.untracked.vstring('/Cosmics/Commissioning09-v3/RAW')
        readFiles.extend( [
                '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/560/DC089E4B-5ED4-DE11-A179-000423D98FBC.root'
        
            ] );

        secFiles.extend([
            ])
    
    elif dataType == 'FileStream' : 
        # data dat
        readFiles.extend( [
                'file:/lookarea_SM/MWGR_29.00105760.0001.A.storageManager.00.0000.dat'
        
            ] );

    process.valGtDigis.AlternativeNrBxBoardDaq = 0x101
    process.valGtDigis.AlternativeNrBxBoardEvm = 0x2

if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/l1GtHwValidation_source.root'

process.l1demon.disableROOToutput = False
process.valGtDigis.RecordLength = cms.vint32(3, 5)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtHwValidation']
process.MessageLogger.categories.append('L1GtHwValidation')

process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.FwkJob.limit = 0
process.MessageLogger.cerr.FwkReport.limit = 0
process.MessageLogger.cerr.FwkSummary.limit = 0

process.MessageLogger.debugs = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtHwValidation = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtHwValidation = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtHwValidation = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.DQMStore.verbose = 0
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'L1TEMU'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True
