from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_pp_on_AA_cff import Run2_2018_pp_on_AA
process = cms.Process("MUTRKDQM", Run2_2018_pp_on_AA)

live=True

offlineTesting=not live


#----------------------------
#### Event Source
#----------------------------
# for live online DQM in P5

if (live):
    process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
elif(offlineTesting):
    process.load("DQM.Integration.config.fileinputsource_cfi")
    
 
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

process.dqmSaver.path = '.'

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)   

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
    process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')



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
process.hltHighLevel.HLTPaths = cms.vstring('HLT*Mu*','HLT_*Physics*')
process.hltHighLevel.andOr = cms.bool(True)
process.hltHighLevel.throw =  cms.bool(False)


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


