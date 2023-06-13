import argparse
import sys

# example: cmsRun L1Trigger/Phase2L1ParticleFlow/test/make_l1ct_patternFiles_cfg.py -- --dumpFilesOFF
# example: cmsRun L1Trigger/Phase2L1ParticleFlow/test/make_l1ct_patternFiles_cfg.py -- --dumpFilesOFF

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Optional parameters')

parser.add_argument("--dumpFilesOFF", help="switch on dump file production", action="store_true", default=False)
parser.add_argument("--patternFilesOFF", help="switch on Layer-1 pattern file production", action="store_true", default=False)

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

if args.dumpFilesOFF:
    print(f'Switching off dump file creation: dumpFilesOFF is {args.dumpFilesOFF}')
if args.patternFilesOFF:
    print(f'Switching off pattern file creation: patternFilesOFF is {args.patternFilesOFF}')


import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("RESP", eras.Phase2C17I13M9)

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
            "drop l1tPFCandidates_*_*_*",
            "drop l1tTkPrimaryVertexs_*_*_*")
)

process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D88_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff') # needed to read HCal TPs
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '125X_mcRun4_realistic_v2', '')

process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_cff')
process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer2EG_cff')
process.load('L1Trigger.L1TTrackMatch.l1tGTTInputProducer_cfi')
process.load('L1Trigger.VertexFinder.l1tVertexProducer_cfi')
process.l1tVertexFinderEmulator = process.l1tVertexProducer.clone()
process.l1tVertexFinderEmulator.VertexReconstruction.Algorithm = "fastHistoEmulation"
process.l1tVertexFinderEmulator.l1TracksInputTag = cms.InputTag("l1tGTTInputProducer", "Level1TTTracksConverted")
from L1Trigger.Phase2L1GMT.gmt_cfi import l1tStandaloneMuons
process.l1tSAMuonsGmt = l1tStandaloneMuons.clone()

from L1Trigger.Phase2L1ParticleFlow.l1SeedConePFJetEmulatorProducer_cfi import l1SeedConePFJetEmulatorProducer
from L1Trigger.Phase2L1ParticleFlow.l1tDeregionizerProducer_cfi import l1tDeregionizerProducer
from L1Trigger.Phase2L1ParticleFlow.l1tJetFileWriter_cfi import l1tSeededConeJetFileWriter
process.l1tLayer2Deregionizer = l1tDeregionizerProducer.clone()
process.l1tLayer2SeedConeJetsCorrected = l1SeedConePFJetEmulatorProducer.clone(L1PFObjects = cms.InputTag('l1tLayer2Deregionizer', 'Puppi'),
                                                                                doCorrections = cms.bool(True),
                                                                                correctorFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root"),
                                                                                correctorDir = cms.string('L1PuppiSC4EmuJets'))
process.l1tLayer2SeedConeJetWriter = l1tSeededConeJetFileWriter.clone(jets = "l1tLayer2SeedConeJetsCorrected")

process.l1tLayer1Barrel9 = process.l1tLayer1Barrel.clone()
process.l1tLayer1Barrel9.puAlgo.nFinalSort = 32
process.l1tLayer1Barrel9.regions[0].etaBoundaries = [ -1.5, -0.5, 0.5, 1.5 ] 
process.l1tLayer1Barrel9.boards=cms.VPSet(
        cms.PSet(
            regions=cms.vuint32(*[0+9*ie+i for ie in range(3) for i in range(3)])),
        cms.PSet(
            regions=cms.vuint32(*[3+9*ie+i for ie in range(3) for i in range(3)])),
        cms.PSet(
            regions=cms.vuint32(*[6+9*ie+i for ie in range(3) for i in range(3)])),
    )

process.l1tLayer1BarrelTDR = process.l1tLayer1Barrel.clone()
process.l1tLayer1BarrelTDR.regionizerAlgo = cms.string("TDR")
process.l1tLayer1BarrelTDR.regionizerAlgoParameters = cms.PSet(
        nTrack = cms.uint32(22),
        nCalo = cms.uint32(15),
        nEmCalo = cms.uint32(12),
        nMu = cms.uint32(2),
        nClocks = cms.uint32(162),
        doSort = cms.bool(False),
        bigRegionEdges = cms.vint32(-560, -80, 400, -560)
    )

process.l1tLayer1BarrelSerenity = process.l1tLayer1Barrel.clone()
process.l1tLayer1BarrelSerenity.regionizerAlgo = "MultififoBarrel"
process.l1tLayer1BarrelSerenity.regionizerAlgoParameters = cms.PSet(
        barrelSetup = cms.string("Full54"),
        useAlsoVtxCoords = cms.bool(True),
        nClocks = cms.uint32(54),
        nHCalLinks = cms.uint32(2),
        nECalLinks = cms.uint32(1),
        nTrack = cms.uint32(22),
        nCalo = cms.uint32(15),
        nEmCalo = cms.uint32(12),
        nMu = cms.uint32(2))
process.l1tLayer1BarrelSerenity.pfAlgoParameters.nTrack = 22
process.l1tLayer1BarrelSerenity.pfAlgoParameters.nSelCalo = 15
process.l1tLayer1BarrelSerenity.pfAlgoParameters.nCalo = 15
process.l1tLayer1BarrelSerenity.pfAlgoParameters.nAllNeutral = 27
process.l1tLayer1BarrelSerenity.puAlgoParameters.nTrack = 22
process.l1tLayer1BarrelSerenity.puAlgoParameters.nIn = 27
process.l1tLayer1BarrelSerenity.puAlgoParameters.nOut = 27
process.l1tLayer1BarrelSerenity.puAlgoParameters.finalSortAlgo = "FoldedHybrid"

from L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_patternWriters_cff import *
if not args.patternFilesOFF:
    process.l1tLayer1Barrel.patternWriters = cms.untracked.VPSet(*barrelWriterConfigs)
    # process.l1tLayer1Barrel9.patternWriters = cms.untracked.VPSet(*barrel9WriterConfigs) # not enabled for now
    process.l1tLayer1HGCal.patternWriters = cms.untracked.VPSet(*hgcalWriterConfigs)
    process.l1tLayer1HGCalElliptic.patternWriters = cms.untracked.VPSet(*hgcalWriterConfigs)
    process.l1tLayer1HGCalNoTK.patternWriters = cms.untracked.VPSet(*hgcalNoTKWriterConfigs)
    process.l1tLayer1HF.patternWriters = cms.untracked.VPSet(*hfWriterConfigs)

process.runPF = cms.Path( 
        process.l1tSAMuonsGmt +
        process.l1tGTTInputProducer +
        process.l1tVertexFinderEmulator +
        process.l1tLayer1Barrel +
        process.l1tLayer1BarrelTDR +
        process.l1tLayer1BarrelSerenity +
        process.l1tLayer1Barrel9 +
        process.l1tLayer1HGCal +
        process.l1tLayer1HGCalElliptic +
        process.l1tLayer1HGCalNoTK +
        process.l1tLayer1HF +
        process.l1tLayer1 +
        process.l1tLayer2Deregionizer +
        process.l1tLayer2SeedConeJetsCorrected +
        # process.l1tLayer2SeedConeJetWriter +
        process.l1tLayer2EG
    )
process.runPF.associate(process.L1TLayer1TaskInputsTask)


#####################################################################################################################
## Layer 2 e/gamma 

if not args.patternFilesOFF:
    process.l1tLayer2EG.writeInPattern = True
    process.l1tLayer2EG.writeOutPattern = True
    process.l1tLayer2EG.inPatternFile.maxLinesPerFile = eventsPerFile_*54
    process.l1tLayer2EG.outPatternFile.maxLinesPerFile = eventsPerFile_*54

#####################################################################################################################
## Layer 2 seeded-cone jets 
if not args.patternFilesOFF:
    process.runPF.insert(process.runPF.index(process.l1tLayer2SeedConeJetsCorrected)+1, process.l1tLayer2SeedConeJetWriter)
    process.l1tLayer2SeedConeJetWriter.maxLinesPerFile = eventsPerFile_*54

if not args.dumpFilesOFF:
  for det in "Barrel", "BarrelTDR", "BarrelSerenity", "Barrel9", "HGCal", "HGCalElliptic", "HGCalNoTK", "HF":
        l1pf = getattr(process, 'l1tLayer1'+det)
        l1pf.dumpFileName = cms.untracked.string("TTbar_PU200_"+det+".dump")


process.source.fileNames  = [ '/store/cmst3/group/l1tr/gpetrucc/12_5_X/NewInputs125X/150223/TTbar_PU200/inputs125X_%d.root' % i for i in (1,2,3,4,5) ]

