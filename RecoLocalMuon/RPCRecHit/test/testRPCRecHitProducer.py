import FWCore.ParameterSet.Config as cms

process = cms.Process("RECLUSTERIZATION")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRUZET4_V5P::All"
process.GlobalTag.globaltag = 'START53_V27::All'
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
        '/store/data/Run2012D/RPCMonitor/RAW/v1/000/207/900/1AFE2D3D-A836-E211-B86A-001D09F23D1D.root'        
        #'rfio:/castor/cern.ch/cms/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/063/050/0C9E6E9C-DC84-DD11-A5F7-000423D6CA72.root'
        )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:rechit01.root')
)

process.p = cms.Path(process.rpcRecHits)
process.ep = cms.EndPath(process.out)


