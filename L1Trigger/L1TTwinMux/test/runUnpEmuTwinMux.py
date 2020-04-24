##Unpackers and Emulator of BMTF-TwinMux

import FWCore.ParameterSet.Config as cms

process = cms.Process("TwinMuxRawToDigi")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "80X_dataRun2_Express_v10"

process.load("EventFilter.RPCRawToDigi.rpcUnpackingModule_cfi")
process.load("CondTools.RPC.RPCLinkMap_sqlite_cff")
process.load("EventFilter.L1TXRawToDigi.twinMuxStage2Digis_cfi")
process.load("EventFilter.RPCRawToDigi.RPCTwinMuxRawToDigi_cfi")
process.RPCTwinMuxRawToDigi.bxMin = cms.int32(-5)
process.RPCTwinMuxRawToDigi.bxMax = cms.int32(5)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )


process.source = cms.Source ("NewEventStreamFileReader",  #"PoolSource",
#process.source = cms.Source ("PoolSource",
 
       fileNames=cms.untracked.vstring(
'/store/t0streamer/Data/PhysicsMuons/000/280/307/run280307_ls0016_streamPhysicsMuons_StorageManager.dat',

        ),
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


####Event Setup Producer
process.load('L1Trigger.L1TTwinMux.fakeTwinMuxParams_cff')

process.esProd = cms.EDAnalyzer("EventSetupRecordDataGetter",
   toGet = cms.VPSet(
      cms.PSet(record = cms.string('L1TwinMuxParamsRcd'),
               data = cms.vstring('L1TwinMuxParams'))      
              ),
   verbose = cms.untracked.bool(True)
)

###TwinMux Emulator
process.load('L1Trigger.L1TTwinMux.simTwinMuxDigis_cfi')
process.simTwinMuxDigis.DTDigi_Source = cms.InputTag("twinMuxStage2Digis:PhIn")
process.simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag("twinMuxStage2Digis:ThIn")
process.simTwinMuxDigis.RPC_Source = cms.InputTag("RPCTwinMuxRawToDigi")


process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")




#########################################
##########Ntuples Block#################
########################################
#process.load("UIoannina.TrUpS.L1TwinMuxProducer")
#
#process.L1TwinMuxProducer = process.L1TwinMuxProducer.clone(
#     twinmuxOutputDigis = cms.InputTag("twinMuxStage2Digis:PhOut"),
#     twinmuxInputPhDigis = cms.InputTag("twinMuxStage2Digis:PhIn"),
#       twinmuxInputThDigis = cms.InputTag("twinMuxStage2Digis:ThIn"),
#     twinmuxInputRPCDigis = cms.InputTag("RPCTwinMuxRawToDigi")
#
#)
#
#process.L1TwinMuxProducerEmulator = process.L1TwinMuxProducer.clone(
#     twinmuxOutputDigis = cms.InputTag("simTwinMuxDigis"),
#     twinmuxInputPhDigis = cms.InputTag("twinMuxStage2Digis:PhIn"),
#        twinmuxInputThDigis = cms.InputTag("twinMuxStage2Digis:ThIn"),
#     twinmuxInputRPCDigis = cms.InputTag("RPCTwinMuxRawToDigi")
#
#)
#process.load("UIoannina.TrUpS.EVRProducer_cfi")
#
#
# output file
#process.TFileService = cms.Service("TFileService",
#     fileName = cms.string('Ntuple_l1ttwinmux_data_run280307.root')
#)

############################



process.L1TMuonSeq = cms.Sequence(process.RPCTwinMuxRawToDigi
                     + process.twinMuxStage2Digis
                     + process.rpcUnpackingModule
                     + process.esProd
                     + process.simTwinMuxDigis                     
#                     + process.EVRTProducer
#                     + process.L1TwinMuxProducer    
#                     + process.L1TwinMuxProducerEmulator  
)


process.L1TMuonPath = cms.Path(process.L1TMuonSeq)

 process.out = cms.OutputModule("PoolOutputModule", 
     outputCommands = cms.untracked.vstring(
         'drop *',
         #'keep *CSC*_*_*_*',
         'keep *RPC*_*_*_*',
         'keep *DT*_*_*_*',
         'keep *L1Mu*_*_*_*',
         'keep *_*Muon*_*_*',
         'keep *_*gen*_*_*',
         'keep *_*TwinMux*_*_*',
         'keep *_*Bmtf*_*_*',
         'keep GenEventInfoProduct_generator_*_*'),
 
   fileName = cms.untracked.string("l1ttwinmux.root")
 )

process.output_step = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.L1TMuonPath)
process.schedule.extend([process.output_step])
