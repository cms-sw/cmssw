import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("RESP", eras.Phase2C9)

process.load('Configuration.StandardSequences.Services_cff')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
            wantSummary = cms.untracked.bool(True),
            numberOfThreads = cms.untracked.uint32(2),
            numberOfStreams = cms.untracked.uint32(1),
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/Phase2HLTTDRWinter20DIGI/TT_TuneCP5_14TeV-powheg-pythia8/GEN-SIM-DIGI-RAW/PU200_110X_mcRun4_realistic_v3-v2/110000/005E74D6-B50E-674E-89E6-EAA9A617B476.root',)
)

process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '123X_mcRun4_realistic_v3', '')

process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff') # needed to read HCal TPs
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff") 
process.load("L1Trigger.TrackerDTC.ProducerES_cff") 
process.load("L1Trigger.TrackerDTC.ProducerED_cff") 
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

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

process.PFInputsTask = cms.Task(
    process.TTClustersFromPhase2TrackerDigis,
    process.TTStubsFromPhase2TrackerDigis,
    process.TrackerDTCProducer,
    process.offlineBeamSpot,
    process.l1tTTTracksFromTrackletEmulation,
    process.SimL1EmulatorTask
)
process.runPF = cms.Path( 
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
process.runPF.associate(process.PFInputsTask)
process.schedule = cms.Schedule(process.runPF)


#####################################################################################################################
## Layer 2 e/gamma 

process.l1tLayer2EG.writeInPattern = True
process.l1tLayer2EG.writeOutPattern = True
process.l1tLayer2EG.inPatternFile.maxLinesPerFile = eventsPerFile_*54
process.l1tLayer2EG.outPatternFile.maxLinesPerFile = eventsPerFile_*54

#####################################################################################################################
## Layer 2 seeded-cone jets 
process.l1tLayer2SeedConeJetWriter.maxLinesPerFile = eventsPerFile_*54
