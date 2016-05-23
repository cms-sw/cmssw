import FWCore.ParameterSet.Config as cms

process = cms.Process("HARVESTING")

#----------------------------
#### Histograms Source
#----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.pbsource_cfi")
process.source.loadFiles = cms.untracked.bool(False)
process.source.streamLabel = cms.untracked.string("streamDQMPLT")
process.source.nextLumiTimeoutMillis = cms.untracked.int32(500)

#----------------------------
#### DQM Environment
#----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'BRIL'
process.dqmEnv.eventInfoFolder = 'EventInfo'
process.dqmSaver.tag = 'BRIL'
#-----------------------------

if process.dqmRunConfig.type.value() == "production":
    rid = process.source.runInputDir.value()
    process.source.runInputDir = rid + ":" + "/cmsnfsscratch/globalscratch/cmsbril/PLT/DQM/"
    
    print "Modified input source:", process.source

# remove EventInfo
process.dqmEnv.eventInfoFolder = 'EventInfo/Random'

process.BrilClient = cms.EDAnalyzer("BrilClient")

process.bril_path = cms.Path(process.BrilClient)
process.p = cms.EndPath(process.dqmEnv + process.dqmSaver)

process.schedule = cms.Schedule(process.bril_path, process.p)
