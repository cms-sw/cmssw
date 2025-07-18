"""
example run scenario:

cmsRun global_reco_PPS_test_cfg.py globalTag=auto:run3_data_prompt inputFiles=file:run_xyz.root maxEvents=100 dqm=1

"""

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3
import FWCore.ParameterSet.VarParsing as VarParsing

#This is data-file from run 381380 which contains standard PPS data, excluding silicon strips
default_input = '/store/data/Run2024E/ZeroBias/AOD/PromptReco-v1/000/381/380/00000/cc9dc36a-15d9-430b-9494-058589e42cf9.root'

#This is data-file from run 378869 which contains silicon strips
default_strips = '/store/data/Run2024A/ZeroBias/AOD/PromptReco-v1/000/378/869/00000/2ad791a1-c074-4129-a220-704ffec6e608.root'

process = cms.Process('RECODQM', Run3)

options = VarParsing.VarParsing()

options.register('globalTag',
                     'auto:run3_data_prompt', 
                      VarParsing.VarParsing.multiplicity.singleton,
                      VarParsing.VarParsing.varType.string,
                      "Global Tag")
                      
options.register('inputFiles',
                 '',
                 VarParsing.VarParsing.multiplicity.list,
                 VarParsing.VarParsing.varType.string)

options.register('maxEvents',
                 1000,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int)

options.register('dqm',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int)  

options.register('strips',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int)                 

options.parseArguments()

fileList = [f'file:{f}' if not (f.startswith('/store/') or f.startswith('file:') or f.startswith('root:')) else f for f in options.inputFiles]
if len(fileList)==0:
    if options.strips:
        fileList.append(default_strips)
    else:
        fileList.append(default_input)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents))
process.verbosity = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    )
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")
process.ctppsDiamondDQMSource.excludeMultipleHits = cms.bool(True)
process.ctppsDiamondDQMSource.plotOnline = cms.untracked.bool(True)
process.ctppsDiamondDQMSource.plotOffline = cms.untracked.bool(False)

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"

# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        fileList
    ),
)


from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag)

# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")


process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP"),

    rpId_45_F = cms.uint32(23),
    rpId_45_N = cms.uint32(3),
    rpId_56_N = cms.uint32(103),
    rpId_56_F = cms.uint32(123),

    outputFile = cms.string("reco_protons_hist.root"),
)

process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),

    rpId_45_F = cms.uint32(23),
    rpId_45_N = cms.uint32(3),
    rpId_56_N = cms.uint32(103),
    rpId_56_F = cms.uint32(123),

    outputFile = cms.string("reco_tracks_hist.root"),
)

process.path = cms.Path(
    process.recoCTPPS
)

process.end_path = cms.EndPath(
    process.ctppsTrackDistributionPlotter
) 

if not options.strips:
    process.end_path *= process.ctppsProtonReconstructionPlotter

if options.dqm:
    process.path *= process.ctppsDQMOfflineSource * process.ctppsDQMOfflineHarvest
    process.end_path *= process.dqmEnv * process.dqmSaver

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
