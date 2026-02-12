import FWCore.ParameterSet.Config as cms

import sys
if 'runkey=hi_run' in sys.argv:
  from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_cff import Run3_pp_on_PbPb_approxSiStripClusters
  process = cms.Process("DQM", Run3_pp_on_PbPb_approxSiStripClusters)
else:
  from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
  process = cms.Process("DQM", Run3_2025)

unitTest = False
if 'unitTest=True' in sys.argv:
	unitTest=True

if unitTest:
  process.load("DQM.Integration.config.unittestinputsource_cfi")
  from DQM.Integration.config.unittestinputsource_cfi import options
else:
  # for live online DQM in P5
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options

  if not options.inputFiles:
      process.source.streamLabel = "streamDQMTestDataScouting"

# for testing in lxplus
#process.load("DQM.Integration.config.fileinputsource_cfi")
#from DQM.Integration.config.fileinputsource_cfi import options

#process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
#)

# import beamspot
from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer as _onlineBeamSpotProducer
process.hltOnlineBeamSpot = _onlineBeamSpotProducer.clone()


process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = 'NGT'
process.dqmSaver.tag = 'NGT'
process.dqmSaver.runNumber = options.runNumber
# process.dqmSaverPB.tag = 'NGT'
# process.dqmSaverPB.runNumber = options.runNumber

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

#---- for P5 (online) DB access
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

### for pp collisions
process.load("DQM.HLTEvF.ScoutingCollectionMonitor_cfi")
process.scoutingCollectionMonitor.topfoldername = "NGT/ScoutingOnline/ScoutingCollections"
process.scoutingCollectionMonitor.onlyScouting = False
process.scoutingCollectionMonitor.onlineMetaDataDigis = "hltOnlineMetaDataDigis"
process.scoutingCollectionMonitor.rho = ["hltScoutingPFPacker", "rho"]

process.load("DQM.HLTEvF.ScoutingDileptonMonitor_cfi")
process.ScoutingDileptonMonitorOnline.OutputInternalPath = "NGT/ScoutingOnline/DiLepton"

process.dqmcommon = cms.Sequence(process.dqmEnv
                               * process.dqmSaver)#*process.dqmSaverPB)

process.p = cms.Path(process.dqmcommon *
                     process.hltOnlineBeamSpot *
                     process.scoutingCollectionMonitor *
                     process.ScoutingDileptonMonitorOnline
                     )

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
print("Global Tag used:", process.GlobalTag.globaltag.value())
print("Final Source settings:", process.source)
