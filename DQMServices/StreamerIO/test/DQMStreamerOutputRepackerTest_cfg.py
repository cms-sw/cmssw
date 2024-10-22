from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("daqstreamer")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring("DQMStreamerOutputModule", "DQMStreamerOutputRepacker"),

    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING')),
    destinations = cms.untracked.vstring('cout')
)


from Configuration.Applications.ConfigBuilder import filesFromDASQuery 

#dataset = "/RelValTTbar_13/CMSSW_8_1_0-PU25ns_81X_upgrade2017_realistic_v26_HLT2017-v1/GEN-SIM-DIGI-RAW"
#dataset = "/StreamExpressPA/PARun2016C-PromptCalibProdSiStripGains-Express-v1/ALCAPROMPT"
#dataset = "/RelValZMM_13/CMSSW_9_0_0_pre2-PU25ns_90X_mcRun2_asymptotic_v0-v1/GEN-SIM-DIGI-RAW-HLTDEBUG"
#dataset = "/SinglePhoton/Run2016C-v2/RAW"
#read, sec = filesFromDASQuery("file dataset=%s" % dataset, option=" --limit 10000 ")
read, sec = ["file:%s" % sys.argv[1]], []

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(read),
    secondaryFileNames = cms.untracked.vstring(sec),
)
print("Selected %d files.", process.source)

process.poolOutput = cms.OutputModule('DQMStreamerOutputRepackerTest')
process.output = cms.EndPath(process.poolOutput)

print(process.source) 
