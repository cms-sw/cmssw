import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

produceEGStage2Pattern = False

process = cms.Process("RESP", eras.Phase2C9)

process.load('Configuration.StandardSequences.Services_cff')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True), allowUnscheduled = cms.untracked.bool(False) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:inputs110X.root'),
    inputCommands = cms.untracked.vstring("keep *", 
            "drop l1tPFClusters_*_*_*",
            "drop l1tPFTracks_*_*_*",
            "drop l1tPFCandidates_*_*_*")
)

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff') # needed to read HCal TPs
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '123X_mcRun4_realistic_v3', '')

process.load("L1Trigger.Phase2L1ParticleFlow.l1ParticleFlow_cff")
process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_cff')
process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer2EG_cff')
process.load('L1Trigger.L1TTrackMatch.L1GTTInputProducer_cfi')
process.load('L1Trigger.VertexFinder.VertexProducer_cff')
process.L1VertexFinderEmulator = process.VertexProducer.clone()
process.L1VertexFinderEmulator.VertexReconstruction.Algorithm = "fastHistoEmulation"
process.L1VertexFinderEmulator.l1TracksInputTag = cms.InputTag("L1GTTInputProducer", "Level1TTTracksConverted")
from L1Trigger.Phase2L1GMT.gmt_cfi import standaloneMuons
process.L1SAMuonsGmt = standaloneMuons.clone()

process.l1ctLayer1Barrel9 = process.l1ctLayer1Barrel.clone()
process.l1ctLayer1Barrel9.regions[0].etaBoundaries = [ -1.5, -0.5, 0.5, 1.5 ] 
process.l1ctLayer1Barrel9.boards=cms.VPSet(
        cms.PSet(
            regions=cms.vuint32(list(range(0, 3)) + [x+9 for x in range(0, 3)] + [x+18 for x in range(0, 3)])),
        cms.PSet(
            regions=cms.vuint32(list(range(3, 6)) + [x+9 for x in range(3, 6)] + [x+18 for x in range(3, 6)])),
        cms.PSet(
            regions=cms.vuint32(list(range(6, 9)) + [x+9 for x in range(6, 9)] + [x+18 for x in range(6, 9)])),
    )

process.runPF = cms.Path( 
        process.L1SAMuonsGmt +
        process.L1GTTInputProducer +
        process.L1VertexFinderEmulator +
        process.pfTracksFromL1Tracks +
        process.l1ParticleFlow_calo +
        process.l1ctLayer1Barrel +
        process.l1ctLayer1Barrel9 +
        process.l1ctLayer1HGCal +
        process.l1ctLayer1HGCalNoTK +
        process.l1ctLayer1HF +
        process.l1ctLayer1 +
        process.l1ctLayer2EG
    )

if produceEGStage2Pattern:
    process.l1ctLayer2EG.writeInPattern = True
    process.l1ctLayer2EG.writeOutPattern = True

process.source.fileNames  = [ '/store/cmst3/group/l1tr/gpetrucc/11_1_0/NewInputs110X/110121.done/TTbar_PU200/inputs110X_%d.root' % i for i in (1,3,7,8,9) ]
process.pfClustersFromCombinedCaloHCal.phase2barrelCaloTowers = [cms.InputTag("L1EGammaClusterEmuProducer",)]


for det in "Barrel", "Barrel9", "HGCal", "HGCalNoTK", "HF":
    l1pf = getattr(process, 'l1ctLayer1'+det)
    l1pf.dumpFileName = cms.untracked.string("TTbar_PU200_110X_"+det+".dump")
