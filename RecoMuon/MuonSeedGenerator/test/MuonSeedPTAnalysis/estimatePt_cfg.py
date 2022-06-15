import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

#process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# if the data file doesn't have reco
#process.load("Configuration.StandardSequences.RawToDigi_cff")
#process.load("Configuration.StandardSequences.Reconstruction_cff")

process.GlobalTag.globaltag = "MC_31X_V2::All"

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(False),
    debugVebosity = cms.untracked.uint32(20),
    fileNames = cms.untracked.vstring(
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu10_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu10_1_2.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu10_1_3.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu20_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu20_1_2.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu20_1_3.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu50_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu50_1_2.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu50_1_3.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu100_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu100_1_2.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu100_1_3.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu150_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu150_1_2.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu150_1_3.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu200_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu200_1_2.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu200_1_3.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu350_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu350_1_2.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu500_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu500_1_2.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu1000_1_1.root',
'dcache:/pnfs/cms/WAX/resilient/sckao/CrabProd/mu1000_1_2.root'
), duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

process.getpt = cms.EDAnalyzer("MuonSeedParametrization",
    debug = cms.untracked.bool(False),
    scale = cms.untracked.bool(False),

    rootFileName = cms.untracked.string('seed31x.root'),

    #minCSCHitsPerSegment = cms.int32(5),
    recHitLabel = cms.untracked.string("csc2DRecHits"),
    cscSegmentLabel = cms.untracked.string("cscSegments"),
    #DebugMuonSeed = cms.bool(False),
    #EnableDTMeasurement = cms.bool(True),
    dtrecHitLabel = cms.untracked.string('dt1DRecHits'),
    dtSegmentLabel = cms.untracked.string("dt4DSegments"),
    simHitLabel = cms.untracked.string('g4SimHits'),
    simTrackLabel = cms.untracked.string('g4SimHits'),
    #EnableCSCMeasurement = cms.bool(True)

)


process.dump = cms.EDAnalyzer("EventContentAnalyzer")
#process.p = cms.Path(process.muonCSCDigis+process.muonDTDigis+process.muonRPCDigis + process.muonlocalreco+process.getpt)
process.p = cms.Path(process.getpt)


