import FWCore.ParameterSet.Config as cms

process = cms.Process("daqstreamer")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring("DQMStreamerOutputModule", "DQMStreamerOutputRepacker"),

    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING')),
    destinations = cms.untracked.vstring('cout')
)


from Configuration.Applications.ConfigBuilder import filesFromDASQuery 

dataset = "/RelValTTbar_13/CMSSW_8_1_0-PU25ns_81X_upgrade2017_realistic_v26_HLT2017-v1/GEN-SIM-DIGI-RAW"
#dataset = "/StreamExpressPA/PARun2016C-PromptCalibProdSiStripGains-Express-v1/ALCAPROMPT"
#dataset = "/RelValZMM_13/CMSSW_9_0_0_pre2-PU25ns_90X_mcRun2_asymptotic_v0-v1/GEN-SIM-DIGI-RAW-HLTDEBUG"
#dataset = "/SinglePhoton/Run2016C-v2/RAW"

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
#read, sec = filesFromDASQuery("file dataset=%s" % dataset, option=" --limit 10000 ")
read, sec = ["file:087DC891-28BF-E611-9909-0025905A6088.root"], []
readFiles.extend(read)
secFiles.extend(sec)

print "Selected %d files." % len(readFiles)
process.source = cms.Source("PoolSource",
    fileNames = readFiles,
    secondaryFileNames = secFiles,
)

process.poolOutput = cms.OutputModule('DQMStreamerOutputRepackerTest',
    outputPath = cms.untracked.string("./output/"),
    streamLabel = cms.untracked.string("DQM"),
    runNumber = cms.untracked.uint32(15),
    eventsPerFile = cms.untracked.uint32(10),
)
process.output = cms.EndPath(process.poolOutput)

print process.source 
