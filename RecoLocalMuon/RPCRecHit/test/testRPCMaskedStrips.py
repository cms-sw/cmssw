import FWCore.ParameterSet.Config as cms

process = cms.Process("RECLUSTERIZATION")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V5P::All"
process.prefer("GlobalTag")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")


process.load("RecoLocalMuon.RPCRecHit.rpcRecHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(300)
)

process.source = cms.Source("PoolSource",

    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/BeamCommissioning08/BeamHalo/RECO/v1/000/063/463/0229F62C-2387-DD11-97B2-000423D987FC.root')
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RECO/v1/000/066/722/001B1910-539D-DD11-AD9E-000423D94E70.root')
)


process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:/tmp/sanabria/rechit01.root'),
    outputCommands = cms.untracked.vstring('keep *',
                                           'drop RPCDetIdRPCRecHitsOwnedRangeMap_rpcRecHits__Rec*')
                               
)


process.p = cms.Path(process.rpcRecHits)
process.ep = cms.EndPath(process.out)


