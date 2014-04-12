import FWCore.ParameterSet.Config as cms

process = cms.Process("RECLUSTERIZATION")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.globaltag = 'GR_R_62_V2::All'
process.prefer("GlobalTag")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")


process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(30000)
)

process.source = cms.Source("PoolSource",
                            #    debugFlag = cms.untracked.bool(True),
                            #    debugVebosity = cms.untracked.uint32(10),
                            fileNames = cms.untracked.vstring(
        #'/store/data/Run2012D/RPCMonitor/RAW/v1/000/207/900/1AFE2D3D-A836-E211-B86A-001D09F23D1D.root'        
        #'/store/relval/CMSSW_7_0_0_pre9/SingleMu/RAW/PRE_P62_V8_gedEG_RelVal_mu2011B-v2/00000/D0DF8456-425C-E311-8403-0025905A48F2.root'
        #'/store/relval/CMSSW_7_0_0_pre9/SingleMu/ALCARECO/TkAlMinBias-PRE_62_V8_RelVal_mu2012D-v4/00000/CE4357C2-8C5D-E311-80B9-0025905A608C.root'
        #'/store/relval/CMSSW_7_0_0_pre9/RelValSingleMuPt100/GEN-SIM-DIGI-RECO/START70_V2_gedEG_FastSim-v2/00000/D2794D4E-0E5C-E311-9B59-0025905A60AA.root'
        'file:step2.root'
        #'rfio:/castor/cern.ch/cms/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/063/050/0C9E6E9C-DC84-DD11-A5F7-000423D6CA72.root'
        )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:rechit01.root')
)

process.p = cms.Path(process.rpcRecHits)
process.ep = cms.EndPath(process.out)


