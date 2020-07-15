from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_pp_on_AA_cff import Run2_2018_pp_on_AA
process = cms.Process("MUTRKDQM", Run2_2018_pp_on_AA)



# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('root://cmsxrootd.fnal.gov///store/data/Commissioning2018/Cosmics/RAW/v1/000/308/409/00000/8806755C-FC04-E811-9EAC-02163E0137B4.root'),
#file:/eos/cms/store/express/Commissioning2019/ExpressCosmics/FEVT/Express-v1/000/331/571/00000/35501AC0-29E7-EA4C-AC1C-194D9B2F12D9.root'),
    secondaryFileNames = cms.untracked.vstring()
)

    

#----------------------------
#### DQM Environment
#----------------------------

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.config.environment_cfi")

#----------------------------
# DQM Live Environment
#-----------------------------

dqmRunConfigDefaults = {
    'userarea': cms.PSet(
        type = cms.untracked.string("userarea"),
        collectorPort = cms.untracked.int32(9190),
        collectorHost = cms.untracked.string('lxplus748'),
    ),
}


dqmRunConfigType = "userarea"
dqmRunConfig = dqmRunConfigDefaults[dqmRunConfigType]


process.load("DQMServices.Core.DQMStore_cfi")

process.DQM = cms.Service("DQM",
                  debug = cms.untracked.bool(False),
                  publishFrequency = cms.untracked.double(5.0),
                  collectorPort = dqmRunConfig.collectorPort,
                  collectorHost = dqmRunConfig.collectorHost,
                  filter = cms.untracked.string(''),
)

process.DQMMonitoringService = cms.Service("DQMMonitoringService")

process.load("DQMServices.Components.DQMEventInfo_cfi")
process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")


#upload should be either a directory or a symlink for dqm gui destination
process.dqmSaver.path = "." 
process.dqmSaver.producer = 'MUTRKDQM'
process.dqmSaver.backupLumiCount = 15

TAG = "Muons"
process.dqmEnv.subSystemFolder =TAG
process.dqmSaver.tag = TAG 



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

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')


# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG'),
)
                                                          )

process.dqmmodules = cms.Sequence(process.dqmEnv + process.dqmSaver)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


# output module
#
process.load("Configuration.EventContent.EventContentCosmics_cff")

process.RECOoutput = cms.OutputModule("PoolOutputModule",
                                      outputCommands = process.RecoMuonRECO.outputCommands,                                                                                 fileName = cms.untracked.string('promptrecoCosmics.root') 
                                  )

process.output = cms.EndPath(process.RECOoutput)



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

process.muSTAreco = cms.Sequence(process.CosmicMuonSeed*cosmicMuons)


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

process.allDQM = cms.Sequence(process.muonDQM*process.dqmEnv*process.dqmSaver)


process.allPaths = cms.Path(process.hltHighLevel * process.hltTriggerTypeFilter * process.allReco * process.allDQM)


from DQM.Integration.config.online_customizations_cfi import *

process = customise(process)


#--------------------------
# Service
#--------------------------
process.AdaptorConfig = cms.Service("AdaptorConfig")
