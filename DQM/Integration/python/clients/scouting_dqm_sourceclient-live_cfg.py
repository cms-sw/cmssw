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
  process.load("DQM.Integration.config.unitteststreamerinputsource_cfi")
  from DQM.Integration.config.unitteststreamerinputsource_cfi import options
  process.source.streamLabel = 'streamDQMOnlineScouting'
else:
  process.load("DQM.Integration.config.inputsource_cfi")
  from DQM.Integration.config.inputsource_cfi import options

  if not options.inputFiles:
      process.source.streamLabel = "streamDQMOnlineScouting"

process.load("DQM.Integration.config.environment_cfi")

process.dqmEnv.subSystemFolder = 'ScoutingDQM'
process.dqmSaver.tag = 'ScoutingDQM'
process.dqmSaver.runNumber = options.runNumber
# process.dqmSaverPB.tag = 'ScoutingDQM'
# process.dqmSaverPB.runNumber = options.runNumber

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

#---- for P5 (online) DB access
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

# import beamspot
from RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi import onlineBeamSpotProducer as _onlineBeamSpotProducer
process.hltOnlineBeamSpot = _onlineBeamSpotProducer.clone()

### for pp collisions
process.load("DQM.HLTEvF.ScoutingCollectionMonitor_cfi")
process.scoutingCollectionMonitor.topfoldername = "HLT/ScoutingOnline/Miscellaneous"
process.scoutingCollectionMonitor.onlyScouting = False # this can flipped due to https://its.cern.ch/jira/browse/CMSHLT-3585
process.scoutingCollectionMonitor.onlineMetaDataDigis = "hltOnlineMetaDataDigis"
process.scoutingCollectionMonitor.rho = ["hltScoutingPFPacker", "rho"]
process.dqmcommon = cms.Sequence(process.dqmEnv
                               * process.dqmSaver)#*process.dqmSaverPB)

process.load("DQM.HLTEvF.ScoutingMuonMonitoring_cff")
process.load("DQM.HLTEvF.ScoutingJetMonitoring_cff")
process.load("DQM.HLTEvF.ScoutingElectronMonitoring_cff")
process.load("DQM.HLTEvF.ScoutingRechitMonitoring_cff")
process.load("DQM.HLTEvF.ScoutingDileptonMonitor_cfi")
## Run-1 L1TGT required by ScoutingJetMonitoring https://github.com/cms-sw/cmssw/blob/master/DQMOffline/JetMET/src/JetAnalyzer.cc#L2603-L2611
process.GlobalTag.toGet.append(
 cms.PSet(
 record = cms.string("L1GtTriggerMenuRcd"),
 tag = cms.string('L1GtTriggerMenu_CRAFT09_hlt'),
 )
)

process.p = cms.Path(process.dqmcommon *
                     process.hltOnlineBeamSpot *
                     process.scoutingCollectionMonitor *
                     process.ScoutingMuonMonitoring *
                     process.ScoutingJetMonitoring *
                     process.ScoutingElectronMonitoring *
                     process.ScoutingRecHitsMonitoring *
                     process.ScoutingDileptonMonitorOnline
                     )

### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
print("Global Tag used:", process.GlobalTag.globaltag.value())
print("Final Source settings:", process.source)
