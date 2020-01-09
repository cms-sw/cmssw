#################################################################
#                                                               #
# RPC Client Configuration file for RPC Source Output Root File #
#                      David Lomidze                            #
#                       INFN Napoli                             #
#                        Feb 2009                               #
#################################################################

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

process = cms.Process("RPCDQMClientTest")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

#process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")

#process.load("Configuration.StandardSequences.MagneticField_cff")

#process.load("CondCore.DBCommon.CondDBSetup_cfi")


##### Run as Emptry Source #######
process.source = cms.Source("EmptySource",
   firstRun = cms.untracked.uint32(70669)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")


################# DQM Client Modules ######################
process.load("DQM.RPCMonitorClient.RPCEventSummary_cfi")
process.rpcEventSummary.EventInfoPath = 'RPC/EventInfo'
process.rpcEventSummary.RPCPrefixDir = 'RPC/RecHits'
process.rpcEventSummary.RPCPrefixDir = 'RPC/RecHits'
process.rpcEventSummary.PrescaleFactor = 1
process.load("DQM.RPCMonitorClient.RPCMon_SS_Dbx_Global_cfi")

################# Quality Tests ############################
from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTesterRPC = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/RPCMonitorClient/test/RPCQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1)
)


################# Open Root file and provide MEs ############
process.ReadMeFromFile = DQMEDHarvester("ReadMeFromFile",
#InputFile = cms.untracked.string('/afs/cern.ch/user/d/dlomidze/scratch0/CMSSW_3_0_0_pre3/src/DQM/RPCMonitorClient/python/DQM_V0001_RPC_R000069800.root')
InputFile = cms.untracked.string('rfio:/castor/cern.ch/user/d/dlomidze/RPC/GlobalRuns/CosmicsCommissioning08-PromptReco-v2RECO/70664/root/Merge_tot.root')
#InputFile = cms.untracked.string('rfio:/castor/cern.ch/user/d/dlomidze/DQM_150.000_RPCEvents.root')                                       
#InputFile = cms.untracked.string('file:/afs/cern.ch/user/d/dlomidze/scratch0/DQM_Merged_V3_R70664.root')
#InputFile = cms.untracked.string('file:/afs/cern.ch/user/d/dlomidze/scratch0/CMSSW_3_1_0_pre2/src/DQM/RPCMonitorDigi/python/DQM_500.000_RPCEvents.root')
)


################# RPC Client Modules #######################
process.RPCDeadChannelTest = DQMEDHarvester("RPCDeadChannelTest")
process.RPCOccupancyTest = DQMEDHarvester("RPCOccupancyTest")
process.RPCClusterSizeTest = DQMEDHarvester("RPCClusterSizeTest")
process.RPCChamberQuality = DQMEDHarvester("RPCChamberQuality")
#process.RPCDCSDataSimulator = DQMEDHarvester("RPCDCSDataSimulator")
process.RPCMultiplicityTest = DQMEDHarvester("RPCMultiplicityTest")
process.RPCOccupancyChipTest = DQMEDHarvester("RPCOccupancyChipTest");
process.RPCNoisyStripTest = DQMEDHarvester("RPCNoisyStripTest");

#process.p = cms.Path(process.ReadMeFromFile*process.qTesterRPC*process.RPCClusterSizeTest*process.RPCDeadChannelTest*process.RPCOccupancyTest*process.RPCDCSDataSimulator*process.RPCMultiplicityTest*process.RPCChamberQuality*process.dqmSaver)

process.p = cms.Path(process.ReadMeFromFile*process.qTesterRPC*process.RPCClusterSizeTest*process.RPCDeadChannelTest*process.RPCOccupancyTest*process.RPCMultiplicityTest*process.RPCOccupancyChipTest*process.RPCNoisyStripTest*process.RPCChamberQuality*process.dqmSaver)



#process.p = cms.Path(process.ReadMeFromFile*process.RPCOccupancyTest*process.dqmSaver)


################ DQM Enviroment ###################
process.dqmEnv.subSystemFolder = 'RPC'

############## DQM Saver ###############
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
#process.dqmSaver.saveByRun = -1
#process.dqmSaver.saveAtJobEnd = True

#dqmSaver = cms.EDFilter("DQMFileSaver",
    # Save file every N runs (-1: disabled)
#    saveByRun = cms.untracked.int32(-1),
    # Save file at the end of the job
 #   saveAtJobEnd = cms.untracked.bool(True)
  #                      )


#process.DQMStore.verbose = 1

######## DQM GUI ########
process.DQM.collectorHost = ''
process.DQM.collectorPort = 9090
process.DQM.debug = False



