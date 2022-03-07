from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("HARVESTING", Run3)

#----------------------------
#### Histograms Source
#----------------------------
# for live online DQM in P5
process.load("DQM.Integration.config.pbsource_cfi")
from DQM.Integration.config.pbsource_cfi import options
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
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'BRIL'
process.dqmSaverPB.runNumber = options.runNumber
#-----------------------------

if process.dqmRunConfig.type.value() == "production":
    rid = process.source.runInputDir.value()
    process.source.runInputDir = rid + ":" + "/cmsnfsscratch/globalscratch/cmsbril/PLT/DQM/"
    
    print("Modified input source:", process.source)

# remove EventInfo
process.dqmEnv.eventInfoFolder = 'EventInfo/Random'

process.BrilClient = DQMEDHarvester("BrilClient")

process.bril_path = cms.Path(process.BrilClient)
process.p = cms.EndPath(process.dqmEnv + process.dqmSaver + process.dqmSaverPB)

process.schedule = cms.Schedule(process.bril_path, process.p)
