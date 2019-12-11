import FWCore.ParameterSet.Config as cms



process = cms.Process('ReDoRecHit')

#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')



process.maxEvents = cms.untracked.PSet(
                       input = cms.untracked.int32(300),
                      
                    )

process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")


process.load('Configuration.Geometry.GeometryExtended2023D41Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D41_cff')

process.load('RecoLocalMuon.Configuration.RecoLocalMuon_cff')
process.dt1DRecHits.dtDigiLabel = cms.InputTag('simMuonDTDigis')
process.rpcRecHits.rpcDigiLabel = cms.InputTag('simMuonRPCDigis')
process.csc2DRecHits.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")


process.rpcRecHits = cms.EDProducer("RPCRecHitProducer",
               recAlgoConfig = cms.PSet(
                                 iRPCConfig = cms.PSet( # iRPC Algorithm
                                   useAlgorithm = cms.bool(False), # useIRPCAlgorithm: if true - use iRPC Algorithm;
                                   returnOnlyAND = cms.bool(True), # returnOnlyAND: if true algorithm will return only associated clusters;
                                   thrTimeHR = cms.double(2), # [ns] thrTimeHR: threshold for high radius clustering;
                                   thrTimeLR = cms.double(2), # [ns] thrTimeLR: threshold for low radius clustering;
                                   thrDeltaTimeMin = cms.double(-30), # [ns] thrDeltaTimeMin: min delta time threshold for association clusters between HR and LR;
                                   thrDeltaTimeMax = cms.double(30), # [ns] thrDeltaTimeMax: max delta time threshold for association clusters between HR and LR;
                                   signalSpeed = cms.double(19.786302) # [cm/ns] signalSpeed: speed of signal inside strip.
                                 ),
                               ),
               recAlgo = cms.string('RPCRecHitStandardAlgo'),
               rpcDigiLabel = cms.InputTag('simMuonRPCDigis'),
               maskSource = cms.string('File'),
               maskvecfile = cms.FileInPath('RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat'),
               deadSource = cms.string('File'),
               deadvecfile = cms.FileInPath('RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat')
             )

# Input source
process.source = cms.Source('PoolSource',
                  fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/k/kshchabl/public/025D27E1-8363-B54D-8E59-15E4E4D8D0A4.root')
                 )

# Output
process.out = cms.OutputModule("PoolOutputModule",
                fileName= cms.untracked.string('MC_RPC_RecHit_2.root'),
                outputCommands = cms.untracked.vstring('drop *',
                "keep *RPC*_*_*_*",
                "keep *_standAloneMuons_*_*",
                "keep *_csc*_*_*",
                "keep *_dt_*_*",
                "keep *ME0*_*_*_*",
                "keep *GEM*_*_*_*")
                 )

    


process.p = cms.Path( process.rpcRecHits + (process.dt1DRecHits * process.dt4DSegments)+  (process.csc2DRecHits * process.cscSegments)  )
process.e = cms.EndPath(process.out)
