#
# cfg file to pack a GT EVM record
#
# V M Ghete 2009-04-06

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestL1GtEvmPacker')

###################### user choices ######################


# choose the type of sample used (True for RelVal, False for data)
useRelValSample = True 
#useRelValSample=False 

# actual GlobalTag must be appropriate for the sample used

if useRelValSample == True :
    useGlobalTag = 'IDEAL_V12'
    #useGlobalTag='STARTUP_V9'
else :
    useGlobalTag = 'CRAFT_ALL_V11'

# change to True to use local files
#     the type of file should match the choice of useRelValSample and useGlobalTag
useLocalFiles = False 

###################### end user choices ###################


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(10)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if useRelValSample == True :
    if useGlobalTag.count('IDEAL') :

        #/RelValTTbar/CMSSW_2_2_4_IDEAL_V11_v1/GEN-SIM-DIGI-RAW-HLTDEBUG
        dataset = cms.untracked.vstring('RelValTTbar_CMSSW_2_2_4_IDEAL_V11_v1')
        
        readFiles.extend([
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/02697009-5CF3-DD11-A862-001D09F2423B.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/064657A8-59F3-DD11-ACA5-000423D991F0.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0817F6DE-5BF3-DD11-880D-0019DB29C5FC.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0899697C-5AF3-DD11-9D21-001617DBD472.root'
            ]);


        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        #/RelValTTbar/CMSSW_2_2_4_STARTUP_V8_v1/GEN-SIM-DIGI-RAW-HLTDEBUG
        dataset = cms.untracked.vstring('RelValTTbar_CMSSW_2_2_4_STARTUP_V8_v1')
        
        readFiles.extend([
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/069AA022-5BF3-DD11-9A56-001617E30D12.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/08DA99A6-5AF3-DD11-AAC1-001D09F24493.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0A725E15-5BF3-DD11-8B4B-000423D99CEE.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0AF5B676-5AF3-DD11-A22F-001617DBCF1E.root'
            ]);


        secFiles.extend([
            ])
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    

else : 

    # data
    dataset = '/Cosmics/Commissioning09-v1/RAW'
    print '   Running on set: '+ dataset    
    
    readFiles.extend( [
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/00BD9A1F-B908-DE11-8B2C-000423D94A04.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/025E8B48-B608-DE11-A0EE-00161757BF42.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/027AA271-D208-DE11-9A7F-001617DBD5AC.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/04281D2F-D108-DE11-9A27-000423D944DC.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/065B0C1C-C008-DE11-A32B-001617E30F48.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/08B1054B-BD08-DE11-AF8B-001617C3B78C.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/0C055C33-D108-DE11-B678-001617C3B73A.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/0E480977-D208-DE11-BA78-001617C3B6E2.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/0E79251B-B908-DE11-83FF-000423D99CEE.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/101B8CA0-B508-DE11-B614-000423D99160.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/12C62C71-BF08-DE11-A48C-000423D99614.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/16A77E08-B008-DE11-9121-000423D8F63C.root'
        ]);

    secFiles.extend([
        ])

if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_L1GtEvmPacker_source.root'


# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'
# FIXME remove when Global Tag OK
process.load('L1Trigger.Configuration.L1Trigger_FakeConditions_cff')

# L1 GT EVM packer
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi")

# input tag for GT readout collection: 
# input tag for GMT readout collection: 
#     source        = hardware record

if useRelValSample == True :
    evmGtInputTag = 'simGtDigis'
else :
    evmGtInputTag = 'l1GtEvmUnpack'

process.l1GtEvmPack.EvmGtInputTag = evmGtInputTag

# mask for active boards (actually 16 bits)
#      if bit is zero, the corresponding board will not be packed
#      default: no board masked: ActiveBoardsMask = 0xFFFF

# no board masked (default)
#process.l1GtEvmPack.ActiveBoardsMask = 0xFFFF
    
# GTFE only in the record
#process.l1GtEvmPack.ActiveBoardsMask = 0x0000

# GTFE + TCS 
#process.l1GtEvmPack.ActiveBoardsMask = 0x0001

# GTFE + FDL 
#process.l1GtEvmPack.ActiveBoardsMask = 0x0002
     
# GTFE + TCS + FDL
#process.l1GtEvmPack.ActiveBoardsMask = 0x0003

# set it to verbose
process.l1GtEvmPack.Verbosity = cms.untracked.int32(1)

# path to be run

if useRelValSample == True :
    process.p = cms.Path(process.l1GtEvmPack)
else :
    process.p = cms.Path(process.l1GtEvmPack) # FIXME unpack first raw data

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtEvmPack']
process.MessageLogger.destinations = ['L1GtEvmPacker']
process.MessageLogger.L1GtEvmPacker = cms.untracked.PSet(
    threshold=cms.untracked.string('DEBUG'),
    #threshold = cms.untracked.string('INFO'),
    #threshold = cms.untracked.string('ERROR'),
    DEBUG=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    INFO=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    WARNING=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    ERROR=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    default = cms.untracked.PSet( 
        limit=cms.untracked.int32(-1)  
    )
)

# summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# output 

process.outputL1GtEvmPack = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('L1GtEvmPacker.root'),
    # keep only unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtEvmPack_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtEvmPack)
