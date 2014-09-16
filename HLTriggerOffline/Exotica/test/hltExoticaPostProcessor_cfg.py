import FWCore.ParameterSet.Config as cms
import os,sys

process = cms.Process("EDMtoMEConvert")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000

process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("HLTriggerOffline.Exotica.HLTExoticaPostVal_cff")
#process.load("HLTriggerOffline.Exotica.HLTExoticaQualityTester_cfi")

# Decide input data
myinput   = ""

for i in range(0,len(sys.argv)):
    if str(sys.argv[i])=="_input" and len(sys.argv)>i+1:
        myinput = str(sys.argv[i+1])

myfileNames = cms.untracked.vstring('file:hltExoticaValidator'+myinput+'.root')

print "Using inputs : "
print myfileNames

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = myfileNames
)

process.postprocessor_path = cms.Path(
		process.HLTExoticaPostVal
                #* process.hltExoticaQualityTester
)

process.edmtome_path = cms.Path(process.EDMtoME)
process.dqmsave_path = cms.Path(process.DQMSaver)

process.schedule = cms.Schedule(process.edmtome_path,
                                process.postprocessor_path,
                                process.dqmsave_path)

process.DQMStore.referenceFileName = ''
