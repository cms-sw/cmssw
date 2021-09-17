import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Demo")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "92X_dataRun2_Prompt_v11",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  303014,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,           # string, int, or float
                  "run number")

options.parseArguments()


##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripGain2RescaleByGain1=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripGain2RescaleByGain1  = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')
process.GlobalTag.toGet = cms.VPSet(
    ### N.B. This contains the G1_new (to be used for the rescale)
    cms.PSet(record = cms.string("SiStripApvGain3Rcd"),
             tag = cms.string("G1_new"),
             connect = cms.string("sqlite_file:G1_new.db")
             ),
    ### N.B. This contains the G2_old (to be used for the rescale)
    cms.PSet(record = cms.string("SiStripApvGain2Rcd"),
             tag = cms.string("G2_old"),
             connect = cms.string("sqlite_file:G2_old.db")
             ),
    ### N.B. This contains the G1_old (to be used for the rescale)
    cms.PSet(record = cms.string("SiStripApvGainRcd"),
             tag = cms.string("G1_old"),
             connect = cms.string("sqlite_file:G1_old.db")
             )
    )

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("CondTools.SiStrip.rescaleGain2byGain1_cfi") 

# process.demo = cms.EDAnalyzer('SiStripGain2RescaleByGain1',
#                               Record = cms.untracked.string("SiStripApvGainRcd"),
#                               )

##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:G2_new.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                                     tag = cms.string('G2_new')
                                                                     )
                                                            )
                                          )

process.p = cms.Path(process.rescaleGain2byGain1)
