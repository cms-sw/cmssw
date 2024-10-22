from __future__ import print_function
import FWCore.ParameterSet.Config as cms

import sys
if 'runkey=hi_run' in sys.argv:
  from Configuration.Eras.Era_Run3_pp_on_PbPb_approxSiStripClusters_cff import Run3_pp_on_PbPb_approxSiStripClusters
  process = cms.Process("MUTRKDQM", Run3_pp_on_PbPb_approxSiStripClusters)
else:
  from Configuration.Eras.Era_Run3_cff import Run3
  process = cms.Process("MUTRKDQM", Run3)

live=True
unitTest=False
if 'unitTest=True' in sys.argv:
    live=False
    unitTest=True

offlineTesting=not live

#----------------------------
#### Event Source
#----------------------------
# for live online DQM in P5


if (unitTest):
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options

elif (live):
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options

# for testing in lxplus
elif(offlineTesting):
    process.load("DQM.Integration.config.fileinputsource_cfi")
    from DQM.Integration.config.fileinputsource_cfi import options
    
 
print("Running with run type = ", process.runType.getRunType())

if (process.runType.getRunType() != process.runType.cosmic_run):
    print("MuTracking client runs only in cosmics, disabling")
    

#----------------------------
#### DQM Environment
#----------------------------

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.config.environment_cfi")


#----------------------------
#### DQM Live Environment
#----------------------------
process.dqmEnv.subSystemFolder = 'Muons'
process.dqmSaver.tag = 'Muons'
##?? process.dqmSaver.backupLumiCount = 30

# process.dqmSaver.path = '.'
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'Muons'
# process.dqmSaverPB.path = './pb'
process.dqmSaverPB.runNumber = options.runNumber

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver + process.dqmSaverPB)

# Imports

#-------------------------------------------------                             
# GEOMETRY                                                                    
#-------------------------------------------------  

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-----------------------------                                                  
# Magnetic Field                                                                
#-----------------------------  

process.load("Configuration.StandardSequences.MagneticField_cff")

#-----------------------------                                                  
# Cosmics muon reco sequence                                                    
#-----------------------------  

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
process.load("L1Trigger.Configuration.L1TRawToDigi_cff") 

#-------------------------------------------------                             
# GLOBALTAG                                                                    
#-------------------------------------------------                              
# Condition for P5 cluster                          

if (live):
    process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# Condition for lxplus: change and possibly customise the GT
elif(offlineTesting):
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

### HEAVY ION SETTING
if process.runType.getRunType() == process.runType.hi_run:
    rawDataRepackerLabel = 'rawDataRepacker'
    process.muonCSCDigis.InputObjects = rawDataRepackerLabel
    process.muonDTDigis.inputLabel = rawDataRepackerLabel
    process.muonRPCDigis.InputLabel = rawDataRepackerLabel
    process.muonGEMDigis.InputLabel = rawDataRepackerLabel
    process.twinMuxStage2Digis.DTTM7_FED_Source = rawDataRepackerLabel
    process.bmtfDigis.InputLabel = rawDataRepackerLabel
    process.omtfStage2Digis.inputLabel = rawDataRepackerLabel
    process.emtfStage2Digis.InputLabel = rawDataRepackerLabel
    process.gmtStage2Digis.InputLabel = rawDataRepackerLabel
    process.rpcTwinMuxRawToDigi.inputTag = rawDataRepackerLabel
    process.rpcCPPFRawToDigi.inputTag = rawDataRepackerLabel

#------------------------------------
# Cosmic muons reconstruction modules
#------------------------------------

#1 RAW-TO-DIGI 

process.muRawToDigi = cms.Sequence(process.L1TRawToDigi +
                                   process.muonCSCDigis +
                                   process.muonDTDigis +
                                   process.muonRPCDigis +
                                   process.muonGEMDigis)
                                   


#2 LOCAL RECO
from RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff import *
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import *

process.dtlocalreco = cms.Sequence(dt1DRecHits*dt4DSegments)
process.csclocalreco = cms.Sequence(csc2DRecHits*cscSegments)
process.muLocalRecoCosmics = cms.Sequence(process.dtlocalreco+process.csclocalreco+process.rpcRecHits)


#3 STA RECO 

from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *

##Reco Beam Spot from DB 
from RecoVertex.BeamSpotProducer.BeamSpotFakeParameters_cfi import *
process.beamspot = cms.EDAnalyzer("BeamSpotFromDB")
process.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")
process.beampath = cms.Sequence(process.beamspot+process.offlineBeamSpot)


process.muSTAreco = cms.Sequence(process.CosmicMuonSeed*process.cosmicMuons)





#--------------------------                                                                                                         # Service                                                                                                                           #--------------------------                                                                                                                      
process.AdaptorConfig = cms.Service("AdaptorConfig")                                                                                            

#--------------------------
# Filters
#--------------------------
# HLT Filter
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter = cms.EDFilter("HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32(1)
)


# HLT trigger selection (HLT_ZeroBias)
# modified for 0 Tesla HLT menu (no ZeroBias_*)
process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.hltHighLevel.HLTPaths = ['HLT*Mu*','HLT_*Physics*']
process.hltHighLevel.andOr = True
process.hltHighLevel.throw =  False


#-----------------------------                                                                                                     
# DQM monitor modules
#----------------------------- 

process.load("DQM.MuonMonitor.muonCosmicAnalyzer_cff")
process.muonDQM = cms.Sequence(process.muonCosmicAnalyzer)


#--------------------------
# Scheduling
#--------------------------

process.allReco = cms.Sequence(process.muRawToDigi*process.muLocalRecoCosmics*process.beampath*process.muSTAreco)

process.allPaths = cms.Path(process.hltHighLevel * 
                            process.hltTriggerTypeFilter * 
                            process.allReco * 
                            process.muonDQM * 
                            process.dqmmodules)


from DQM.Integration.config.online_customizations_cfi import *

process = customise(process)
process.options.wantSummary = cms.untracked.bool(True)
print("Final Source settings:", process.source)

