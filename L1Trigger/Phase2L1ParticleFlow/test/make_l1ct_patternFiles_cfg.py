import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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
            "drop l1tPFCandidates_*_*_*",
            "drop l1tTkPrimaryVertexs_*_*_*")
)

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff') # needed to read HCal TPs
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '123X_mcRun4_realistic_v3', '')

process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_cff')
process.load('L1Trigger.Phase2L1ParticleFlow.l1ctLayer2EG_cff')
process.load('L1Trigger.L1TTrackMatch.l1tGTTInputProducer_cfi')
process.load('L1Trigger.VertexFinder.l1tVertexProducer_cfi')
process.l1tVertexFinderEmulator = process.l1tVertexProducer.clone()
process.l1tVertexFinderEmulator.VertexReconstruction.Algorithm = "fastHistoEmulation"
process.l1tVertexFinderEmulator.l1TracksInputTag = cms.InputTag("l1tGTTInputProducer", "Level1TTTracksConverted")
from L1Trigger.Phase2L1GMT.gmt_cfi import l1tStandaloneMuons
process.l1tSAMuonsGmt = l1tStandaloneMuons.clone()

from L1Trigger.Phase2L1ParticleFlow.l1tSeedConePFJetProducer_cfi import l1tSeedConePFJetEmulatorProducer
from L1Trigger.Phase2L1ParticleFlow.l1tDeregionizerProducer_cfi import l1tDeregionizerProducer
from L1Trigger.Phase2L1ParticleFlow.l1tJetFileWriter_cfi import l1tSeededConeJetFileWriter
process.l1tLayer2Deregionizer = l1tDeregionizerProducer.clone()
process.l1tLayer2SeedConeJetsCorrected = l1tSeedConePFJetEmulatorProducer.clone(L1PFObject = ('l1tLayer2Deregionizer', 'Puppi'),
                                                                                doCorrections = True,
                                                                                correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root",
                                                                                correctorDir = "L1PuppiSC4EmuJets")
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

from L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_patternWriters_cff import *
process.l1tLayer1Barrel.patternWriters = cms.untracked.VPSet(*barrelWriterConfigs)
#process.l1tLayer1Barrel9.patternWriters = cms.untracked.VPSet(*barrel9WriterConfigs) # not enabled for now
process.l1tLayer1HGCal.patternWriters = cms.untracked.VPSet(*hgcalWriterConfigs)
process.l1tLayer1HGCalNoTK.patternWriters = cms.untracked.VPSet(*hgcalNoTKWriterConfigs)
process.l1tLayer1HF.patternWriters = cms.untracked.VPSet(*hfWriterConfigs)

process.runPF = cms.Path( 
        process.l1tSAMuonsGmt +
        process.l1tGTTInputProducer +
        process.l1tVertexFinderEmulator +
        process.l1tLayer1Barrel +
        #process.l1tLayer1Barrel9 +
        process.l1tLayer1HGCal +
        process.l1tLayer1HGCalNoTK +
        process.l1tLayer1HF +
        process.l1tLayer1 +
        process.l1tLayer2Deregionizer +
        process.l1tLayer2SeedConeJetsCorrected +
        process.l1tLayer2SeedConeJetWriter +
        process.l1tLayer2EG
    )
process.runPF.associate(process.L1TLayer1TaskInputsTask)


#####################################################################################################################
## Layer 2 e/gamma 

process.l1tLayer2EG.writeInPattern = True
process.l1tLayer2EG.writeOutPattern = True
process.l1tLayer2EG.inPatternFile.maxLinesPerFile = eventsPerFile_*54
process.l1tLayer2EG.outPatternFile.maxLinesPerFile = eventsPerFile_*54

#####################################################################################################################
## Layer 2 seeded-cone jets 
process.l1tLayer2SeedConeJetWriter.maxLinesPerFile = eventsPerFile_*54

process.source.fileNames  = [ '/store/cmst3/group/l1tr/gpetrucc/11_1_0/NewInputs110X/110121.done/TTbar_PU200/inputs110X_%d.root' % i for i in (1,3,7,8,9) ]
process.l1tPFClustersFromL1EGClusters.src = cms.InputTag("L1EGammaClusterEmuProducer",)
process.l1tPFClustersFromCombinedCaloHCal.phase2barrelCaloTowers = [cms.InputTag("L1EGammaClusterEmuProducer",)]
process.l1tPFClustersFromHGC3DClusters.src  = cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering")
process.l1tPFClustersFromCombinedCaloHF.hcalCandidates = [ cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering")]
process.l1tPFTracksFromL1Tracks.L1TrackTag = cms.InputTag("TTTracksFromTrackletEmulation","Level1TTTracks")
process.l1tGTTInputProducer.l1TracksInputTag = cms.InputTag("TTTracksFromTrackletEmulation","Level1TTTracks")

