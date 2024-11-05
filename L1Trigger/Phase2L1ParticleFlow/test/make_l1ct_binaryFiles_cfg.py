import argparse
import sys

# example: cmsRun L1Trigger/Phase2L1ParticleFlow/test/make_l1ct_patternFiles_cfg.py --patternFilesOFF
# example: cmsRun L1Trigger/Phase2L1ParticleFlow/test/make_l1ct_patternFiles_cfg.py --dumpFilesOFF --serenity

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Optional parameters')

parser.add_argument("--dumpFilesOFF", help="switch on dump file production", action="store_true", default=False)
parser.add_argument("--patternFilesOFF", help="switch on Layer-1 pattern file production", action="store_true", default=False)
parser.add_argument("--serenity", help="use Serenity settigns as default everwhere, i.e. also for barrel", action="store_true", default=False)
parser.add_argument("--tm18", help="Add TM18 emulators", action="store_true", default=False)
parser.add_argument("--split18", help="Make 3 TM18 layer 1 pattern files", action="store_true", default=False)

args = parser.parse_args()

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
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1008))
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:inputs110X.root'),
    inputCommands = cms.untracked.vstring("keep *", 
            "drop l1tPFClusters_*_*_*",
            "drop l1tPFTracks_*_*_*",
            "drop l1tPFCandidates_*_*_*",
            "drop l1tTkPrimaryVertexs_*_*_*",
            "drop l1tKMTFTracks_*_*_*"),
    skipEvents = cms.untracked.uint32(0),
)

process.load('Configuration.Geometry.GeometryExtendedRun4D88Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D88_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff') # needed to read HCal TPs
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '125X_mcRun4_realistic_v2', '')

process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_cff')
process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer2EG_cff')
process.load('L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff')
process.load('L1Trigger.L1TTrackMatch.l1tGTTInputProducer_cfi')
process.load('L1Trigger.L1TTrackMatch.l1tTrackSelectionProducer_cfi')
process.l1tTrackSelectionProducer.processSimulatedTracks = False # these would need stubs, and are not used anyway
process.load('L1Trigger.VertexFinder.l1tVertexProducer_cfi')
from L1Trigger.Configuration.SimL1Emulator_cff import l1tSAMuonsGmt
process.l1tSAMuonsGmt = l1tSAMuonsGmt.clone()
from L1Trigger.L1CaloTrigger.l1tPhase2L1CaloEGammaEmulator_cfi import l1tPhase2L1CaloEGammaEmulator
process.l1tPhase2L1CaloEGammaEmulator = l1tPhase2L1CaloEGammaEmulator.clone()
from L1Trigger.L1CaloTrigger.l1tPhase2CaloPFClusterEmulator_cfi import l1tPhase2CaloPFClusterEmulator
process.l1tPhase2CaloPFClusterEmulator = l1tPhase2CaloPFClusterEmulator.clone()

process.L1TInputTask = cms.Task(
    process.l1tSAMuonsGmt,
    process.l1tPhase2L1CaloEGammaEmulator,
    process.l1tPhase2CaloPFClusterEmulator
)


from L1Trigger.Phase2L1ParticleFlow.l1tJetFileWriter_cfi import l1tSeededConeJetFileWriter
l1ctLayer2SCJetsProducts = cms.VPSet([cms.PSet(jets = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulator"),
                                               nJets = cms.uint32(12),
                                               mht  = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulatorMHT"),
                                               nSums = cms.uint32(2)),
                                      cms.PSet(jets = cms.InputTag("l1tSC8PFL1PuppiCorrectedEmulator"),
                                               nJets = cms.uint32(12))
                                      ])
process.l1tLayer2SeedConeJetWriter = l1tSeededConeJetFileWriter.clone(collections = l1ctLayer2SCJetsProducts)

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

if args.serenity:
    process.l1tLayer1.pfProducers[0] = "l1tLayer1BarrelSerenity"
    process.l1tLayer2EG.tkElectrons[1].pfProducer = "l1tLayer1BarrelSerenity:L1TkElePerBoard"
    process.l1tLayer2EG.tkEms[2].pfProducer = "l1tLayer1BarrelSerenity:L1TkEmPerBoard"

from L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_patternWriters_cff import *
from L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_patternWriters_cff import _eventsPerFile
if not args.patternFilesOFF:
    process.l1tLayer1Barrel.patternWriters = cms.untracked.VPSet(*barrelWriterConfigs)
    process.l1tLayer1BarrelSerenity.patternWriters = cms.untracked.VPSet(barrelSerenityVU9PPhi1Config,barrelSerenityVU13PPhi1Config)
    process.l1tLayer1HGCal.patternWriters = cms.untracked.VPSet(*hgcalWriterConfigs)
    process.l1tLayer1HGCalElliptic.patternWriters = cms.untracked.VPSet(*hgcalWriterConfigs)
    process.l1tLayer1HGCalNoTK.patternWriters = cms.untracked.VPSet(*hgcalNoTKWriterConfigs)
    process.l1tLayer1HF.patternWriters = cms.untracked.VPSet(*hfWriterConfigs)

process.runPF = cms.Path( 
        # process.l1tSAMuonsGmt + 
        # process.l1tPhase2L1CaloEGammaEmulator + 
        # process.l1tPhase2CaloPFClusterEmulator +
        process.l1tGTTInputProducer +
        process.l1tTrackSelectionProducer +
        process.l1tVertexFinderEmulator +
        process.l1tLayer1Barrel +
        process.l1tLayer1BarrelTDR +
        process.l1tLayer1BarrelSerenity +
        process.l1tLayer1HGCal +
        process.l1tLayer1HGCalElliptic +
        process.l1tLayer1HGCalNoTK +
        process.l1tLayer1HF +
        process.l1tLayer1 +
        process.l1tLayer2Deregionizer +
        process.l1tSC4PFL1PuppiCorrectedEmulator +
        process.l1tSC4PFL1PuppiCorrectedEmulatorMHT +
        process.l1tSC8PFL1PuppiCorrectedEmulator +
        # process.l1tLayer2SeedConeJetWriter +
        process.l1tLayer2EG
    )
process.runPF.associate(process.L1TInputTask)
process.runPF.associate(process.L1TLayer1TaskInputsTask)

#####################################################################################################################
## Layer 2 e/gamma 

if not args.patternFilesOFF:
    process.l1tLayer2EG.writeInPattern = True
    process.l1tLayer2EG.writeOutPattern = True
    process.l1tLayer2EG.inPatternFile.maxLinesPerFile = _eventsPerFile*54
    process.l1tLayer2EG.outPatternFile.maxLinesPerFile = _eventsPerFile*54

#####################################################################################################################
## Layer 2 seeded-cone jets 
if not args.patternFilesOFF:
    process.runPF.insert(process.runPF.index(process.l1tSC8PFL1PuppiCorrectedEmulator)+1, process.l1tLayer2SeedConeJetWriter)
    process.l1tLayer2SeedConeJetWriter.maxLinesPerFile = _eventsPerFile*54

if not args.dumpFilesOFF:
  for det in "Barrel", "BarrelTDR", "BarrelSerenity", "HGCal", "HGCalElliptic", "HGCalNoTK", "HF":
        l1pf = getattr(process, 'l1tLayer1'+det)
        l1pf.dumpFileName = cms.untracked.string("TTbar_PU200_"+det+".dump")


if args.tm18:
    process.l1tLayer1HGCalTM18 = process.l1tLayer1HGCal.clone()
    process.l1tLayer1HGCalTM18.regionizerAlgo = "BufferedFoldedMultififo"
    process.l1tLayer1HGCalTM18.regionizerAlgoParameters.nClocks = 162
    del process.l1tLayer1HGCalTM18.regionizerAlgoParameters.nEndcaps 
    del process.l1tLayer1HGCalTM18.regionizerAlgoParameters.nTkLinks
    del process.l1tLayer1HGCalTM18.regionizerAlgoParameters.nCaloLinks
    process.l1tLayer1HGCalNoTKTM18 = process.l1tLayer1HGCalNoTK.clone()
    process.l1tLayer1HGCalNoTKTM18.regionizerAlgo = "BufferedFoldedMultififo"
    process.l1tLayer1HGCalNoTKTM18.regionizerAlgoParameters.nClocks = 162
    del process.l1tLayer1HGCalNoTKTM18.regionizerAlgoParameters.nEndcaps 
    del process.l1tLayer1HGCalNoTKTM18.regionizerAlgoParameters.nTkLinks
    del process.l1tLayer1HGCalNoTKTM18.regionizerAlgoParameters.nCaloLinks
    process.l1tLayer1BarrelSerenityTM18 = process.l1tLayer1BarrelSerenity.clone()
    process.l1tLayer1BarrelSerenityTM18.regionizerAlgo = "MiddleBufferMultififo"
    process.l1tLayer1BarrelSerenityTM18.regionizerAlgoParameters = cms.PSet(
        nTrack = process.l1tLayer1BarrelSerenity.regionizerAlgoParameters.nTrack,
        nCalo = process.l1tLayer1BarrelSerenity.regionizerAlgoParameters.nCalo,
        nEmCalo = process.l1tLayer1BarrelSerenity.regionizerAlgoParameters.nEmCalo,
        nMu = process.l1tLayer1BarrelSerenity.regionizerAlgoParameters.nMu,
    )
    process.l1tLayer1BarrelSerenityTM18.boards = cms.VPSet(*[cms.PSet(regions = cms.vuint32(*range(18*i,18*i+18))) for i in range(3)])
    process.runPF.insert(process.runPF.index(process.l1tLayer1HGCal)+1, process.l1tLayer1HGCalTM18)
    process.runPF.insert(process.runPF.index(process.l1tLayer1HGCalNoTK)+1, process.l1tLayer1HGCalNoTKTM18)
    process.runPF.insert(process.runPF.index(process.l1tLayer1BarrelSerenity)+1, process.l1tLayer1BarrelSerenityTM18)
    # FIXME: we need to schedule a new deregionizer for TM18
    process.runPF.insert(process.runPF.index(process.l1tLayer2EG)+1, process.l1tLayer2EGTM18)
    if not args.patternFilesOFF:
        process.l1tLayer1HGCalTM18.patternWriters = cms.untracked.VPSet(*hgcalTM18WriterConfigs)
        process.l1tLayer1HGCalNoTKTM18.patternWriters = cms.untracked.VPSet(hgcalNoTKOutputTM18WriterConfig)
        process.l1tLayer1BarrelSerenityTM18.patternWriters = cms.untracked.VPSet(*barrelSerenityTM18WriterConfigs)
        process.l1tLayer2EGTM18.writeInPattern = True
        process.l1tLayer2EGTM18.writeOutPattern = True
    if not args.dumpFilesOFF:
        for det in "HGCalTM18", "HGCalNoTKTM18", "BarrelSerenityTM18":
                getattr(process, 'l1tLayer1'+det).dumpFileName = cms.untracked.string("TTbar_PU200_"+det+".dump")
    if args.split18 and not args.patternFilesOFF:
        from FWCore.Modules.preScaler_cfi import preScaler
        for tmSlice, psOffset in (0,1), (6,2), (12,0):
            setattr(process, f"preTM{tmSlice}", preScaler.clone(prescaleFactor = 3, prescaleOffset = psOffset))
            for det in "HGCalTM18", "HGCalNoTKTM18", "BarrelSerenityTM18":
                tsmod = getattr(process, 'l1tLayer1'+det).clone()
                tsmod.dumpFileName = cms.untracked.string("")
                setattr(process, f"l1tLayer1{det}TS{tmSlice}", tsmod)
                setattr(process, f"Write_{det}TS{tmSlice}", cms.Path(getattr(process, f"preTM{tmSlice}")+tsmod))
            getattr(process, f'l1tLayer1HGCalTM18TS{tmSlice}').patternWriters = cms.untracked.VPSet(
                hgcalWriterOutputTM18WriterConfig.clone(outputFileName = f"l1HGCalTM18-outputs-ts{tmSlice}"),
                hgcalWriterVU9PTM18WriterConfig.clone(inputFileName = f"l1HGCalTM18-inputs-vu9p-ts{tmSlice}"),
                hgcalWriterVU13PTM18WriterConfig.clone(inputFileName = f"l1HGCalTM18-inputs-vu13p-ts{tmSlice}")
            )
            getattr(process, f'l1tLayer1HGCalNoTKTM18TS{tmSlice}').patternWriters = cms.untracked.VPSet(
                hgcalNoTKOutputTM18WriterConfig.clone(outputFileName = f"l1HGCalTM18-outputs-ts{tmSlice}"),
            )
            getattr(process, f'l1tLayer1BarrelSerenityTM18TS{tmSlice}').patternWriters = cms.untracked.VPSet(
                barrelSerenityOutputTM18WriterConfig.clone(outputFileName = f"l1BarrelSerenityTM18-outputs-ts{tmSlice}"),
                barrelSerenityVU13PTM18WriterConfig.clone(inputFileName = f"l1BarrelSerenityTM18-inputs-vu13p-ts{tmSlice}")
            )        

process.source.fileNames  = [ '/store/cmst3/group/l1tr/cerminar/14_0_X/fpinputs_131X/v3/TTbar_PU200/inputs131X_1.root' ]
