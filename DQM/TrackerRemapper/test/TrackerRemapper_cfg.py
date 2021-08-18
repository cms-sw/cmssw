import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from Configuration.AlCa.GlobalTag import GlobalTag

options = VarParsing.VarParsing("analysis")

MODE_ANALYZE, MODE_REMAP = 0, 1
RECHITS, DIGIS, CLUSTERS = 1, 2, 3

dataSourceDic = { RECHITS : "ALCARECOTkAlMinBias", #"generalTracks",
                  DIGIS   : "siStripDigis",
                  CLUSTERS: "siStripClusters" }

defaultAnylyzeMode = RECHITS #DIGIS # RECHITS

###################### OPTIONS HANDLER

options.register ("opMode",                                  
                  MODE_ANALYZE,               
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.int,         
                  "Operation Mode")   

options.register ("analyzeMode",                                  
                  defaultAnylyzeMode,               
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.int,         
                  "Analyze Mode") 

options.register ("eventLimit",                                  
                  -1,               
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.int,         
                  "Limits Events Processed in Analyze Mode") 

options.register ("inputRootFile",                                  
                  "/store/express/Run2018D/StreamExpressAlignment/ALCARECO/TkAlMinBias-Express-v1/000/324/980/00000/00E8FB8F-D3AB-C442-BCC2-FEEAE63EA711.root",
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,         
                  "Source Data File - either for analyze or remap")   

options.register ("stripHistogram",                                  
                  "TkHMap_NumberValidHits",               
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,         
                  "Strip Detector Histogram to Remap")   

options.register ("src",   
                  dataSourceDic[defaultAnylyzeMode],                                              
                  VarParsing.VarParsing.multiplicity.singleton, 
                  VarParsing.VarParsing.varType.string,         
                  "Collection Source")    #??

options.register ("globalTag",                                  # option name
                  "auto:run2_data",                             # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, bool, int, or float
                  "Global Tag")                                 # ? help ?
                  
options.parseArguments()

######################

process = cms.Process("Demo")

if options.opMode == MODE_ANALYZE:
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.eventLimit) )
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(
            options.inputRootFile
        )
    )
    runNumber = "Analyze"

elif options.opMode == MODE_REMAP:
    process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
    process.source = cms.Source("EmptySource")
    #run number deduction
    runNumber = str(int(options.inputRootFile.split("_")[-1].split(".")[0][1:]))

process.demo = cms.EDAnalyzer('TrackerRemapper',
                              opMode = cms.int32(options.opMode),
                              analyzeMode = cms.int32(options.analyzeMode),
                              #stripRemapFile = cms.string(options.inputRootFile),
                              #stripHistogram = cms.string(options.stripHistogram),
                              #runString = cms.string(runNumber),
                              src = cms.InputTag(options.src),
                              )

process.p = cms.Path(process.demo)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, "")

process.load("CalibTracker.SiStripCommon.TkDetMapESProducer_cfi")
#process.load("DQM.SiStripCommon.TkHistoMap_cff")
#process.TkDetMap = cms.Service("TkDetMap")
#process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

#### Add these lines to produce a tracker map
process.load("DQMServices.Core.DQMStore_cfg")

# Output root file name:
process.TFileService = cms.Service("TFileService", fileName = cms.string('outputStrip.root') )
