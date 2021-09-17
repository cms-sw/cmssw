import FWCore.ParameterSet.Config as cms

process = cms.Process("PATMuon")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'PRE_ST62_V8::All'

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-RECO/PU_PRE_ST62_V8-v2/00000/E03F79C5-A7EC-E211-A92E-003048F00520.root',
    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/DE640769-94EC-E211-ACE9-003048F23D8C.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/D21DB243-94EC-E211-AEED-003048D2BC36.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/BC74017C-94EC-E211-9F4A-BCAEC532971D.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/90485044-94EC-E211-8BC7-003048F0111A.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/88F032AD-94EC-E211-BA22-0030486730E8.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/78F4633D-94EC-E211-A1FE-003048F11D46.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/68ED2290-94EC-E211-A08E-003048F16B8E.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/2EDDA656-94EC-E211-A208-003048F16F46.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/2C1CD01D-94EC-E211-AA58-001E673982E1.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/18B87680-9EEC-E211-AD34-003048F1C410.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/16040F35-94EC-E211-A460-003048F1DBB6.root',
        '/store/relval/CMSSW_6_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_PRE_ST62_V8-v2/00000/04E987D4-94EC-E211-9AA2-003048C9CC70.root',
    ),
)


## IF twoFileSolution == TRUE:  test with TrackingParticles from input secondary files
## IF twoFileSolution == FALSE: test making them from GEN-SIM-RECO
twoFileSolution = False

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.selMuons = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("pt > 15 && isGlobalMuon"),
    filter = cms.bool(True),
)

process.load("MuonAnalysis.MuonAssociators.muonClassificationByHits_cfi")

process.classByHits = process.classByHitsGlb.clone(muons = "selMuons", muonPreselection = "")

if twoFileSolution:
    process.classByHits.trackingParticles = cms.InputTag("mix","MergedTrackTruth")
    process.go = cms.Path(
        process.selMuons    +
        process.classByHits
    )
else:
    del process.source.secondaryFileNames
    process.go = cms.Path(
        process.selMuons    +
        process.trackingParticlesNoSimHits +
        process.classByHits
    )


process.MessageLogger.cerr.MuonMCClassifier = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
)

