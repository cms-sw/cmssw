############################################################
# define basic process
############################################################

import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils
import os

############################################################
# edit options here
############################################################
L1TRK_INST ="MyL1TrackJets" ### if not in input DIGRAW then we make them in the above step
process = cms.Process(L1TRK_INST)

#L1TRKALGO = 'HYBRID'  #baseline, 4par fit
# L1TRKALGO = 'HYBRID_DISPLACED'  #extended, 5par fit
L1TRKALGO = 'HYBRID_PROMPTANDDISP'

DISPLACED = ''

############################################################
# import standard configurations
############################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(0) # default: 0

############################################################
# input and output
############################################################

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

readFiles = cms.untracked.vstring(
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/c6df2819-ed05-4b98-8f92-81b7d1b1092e.root',
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/3f476d95-1ef7-4be6-977b-6bcd1a7c5678.root',
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/68d651da-4cb7-4bf4-b002-66aecc57a2bc.root',
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/db0e0ce2-4c5a-4988-9dbd-52066e40b9d2.root',
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/257a9712-0a96-47b7-897e-f5d980605e46.root',
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/bee31399-8559-4243-b539-cae1ea897def.root',
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/24629540-2377-4168-9ae5-518ddd4c43a9.root',
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/e31ba8f0-332a-4a1a-8bc0-91a12a5fe3db.root',
    '/store/relval/CMSSW_12_3_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_123X_mcRun4_realistic_v3_2026D77PU200-v1/2580000/17902198-4db6-4fcc-9e8c-787991b4db32.root',
)
secFiles = cms.untracked.vstring()

process.source = cms.Source ("PoolSource",
                            fileNames = readFiles,
                            secondaryFileNames = secFiles,
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            )

process.source.inputCommands = cms.untracked.vstring("keep *","drop l1tTkPrimaryVertexs_L1TkPrimaryVertex__*")

process.TFileService = cms.Service("TFileService", fileName = cms.string('GTTObjects_ttbar200PU.root'), closeFileFast = cms.untracked.bool(True))


############################################################
# L1 tracking: remake stubs?
############################################################

process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
from L1Trigger.TrackTrigger.TTStubAlgorithmRegister_cfi import *
process.load("SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff")

from SimTracker.TrackTriggerAssociation.TTClusterAssociation_cfi import *
TTClusterAssociatorFromPixelDigis.digiSimLinks = cms.InputTag("simSiPixelDigis","Tracker")

process.TTClusterStub = cms.Path(process.TrackTriggerClustersStubs)
process.TTClusterStubTruth = cms.Path(process.TrackTriggerAssociatorClustersStubs)


# DTC emulation
process.load('L1Trigger.TrackerDTC.ProducerES_cff')
process.load('L1Trigger.TrackerDTC.ProducerED_cff')
process.dtc = cms.Path(process.TrackerDTCProducer)#*process.TrackerDTCAnalyzer)

process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")
process.load("L1Trigger.L1TTrackMatch.L1TrackSelectionProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TrackJetProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1GTTInputProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TrackJetEmulationProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TrackFastJetProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TrackerEtMissProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TrackerEtMissEmulatorProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TkHTMissProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.L1TkHTMissEmulatorProducer_cfi")
process.load('L1Trigger.VertexFinder.VertexProducer_cff')


############################################################
# Primary vertex
############################################################
process.L1VertexFinder = process.VertexProducer.clone()
process.pPV = cms.Path(process.L1VertexFinder)
process.L1VertexFinderEmulator = process.VertexProducer.clone()
process.L1VertexFinderEmulator.VertexReconstruction.Algorithm = "fastHistoEmulation"
process.L1VertexFinderEmulator.l1TracksInputTag = cms.InputTag("L1GTTInputProducer","Level1TTTracksConverted")
process.pPVemu = cms.Path(process.L1VertexFinderEmulator)

process.L1TrackFastJets.L1PrimaryVertexTag = cms.InputTag("L1VertexFinder", "l1vertices")
process.L1TrackFastJetsExtended.L1PrimaryVertexTag = cms.InputTag("L1VertexFinder", "l1vertices")
process.L1TrackJets.L1PVertexCollection = cms.InputTag("L1VertexFinder", "l1vertices")
process.L1TrackJetsExtended.L1PVertexCollection = cms.InputTag("L1VertexFinder", "l1vertices")
process.L1TrackerEtMiss.L1VertexInputTag = cms.InputTag("L1VertexFinder", "l1vertices")
process.L1TrackerHTMiss.L1VertexInputTag = cms.InputTag("L1VertexFinder", "l1vertices")
process.L1TrackerEtMissExtended.L1VertexInputTag = cms.InputTag("L1VertexFinder", "l1vertices")
process.L1TrackerHTMissExtended.L1VertexInputTag = cms.InputTag("L1VertexFinder", "l1vertices")
process.L1TrackerEmuEtMiss.L1VertexInputTag = cms.InputTag("L1VertexFinderEmulator", "l1verticesEmulation")


# HYBRID: prompt tracking
if (L1TRKALGO == 'HYBRID'):
    process.TTTracksEmu = cms.Path(process.L1HybridTracks)
    process.TTTracksEmuWithTruth = cms.Path(process.L1HybridTracksWithAssociators)
    process.pL1TrackSelection = cms.Path(process.L1TrackSelectionProducer)
    process.pL1TrackJets = cms.Path(process.L1TrackJets)
    process.pL1TrackFastJets=cms.Path(process.L1TrackFastJets)
    process.pL1GTTInput = cms.Path(process.L1GTTInputProducer)
    process.pL1TrackJetsEmu = cms.Path(process.L1TrackJetsEmulation)
    process.pTkMET = cms.Path(process.L1TrackerEtMiss)
    process.pTkMETEmu = cms.Path(process.L1TrackerEmuEtMiss)
    process.pTkMHT = cms.Path(process.L1TrackerHTMiss)
    process.pTkMHTEmulator = cms.Path(process.L1TrackerEmuHTMiss)
    DISPLACED = 'Prompt'

# HYBRID: extended tracking
elif (L1TRKALGO == 'HYBRID_DISPLACED'):
    process.TTTracksEmu = cms.Path(process.L1ExtendedHybridTracks)
    process.TTTracksEmuWithTruth = cms.Path(process.L1ExtendedHybridTracksWithAssociators)
    process.pL1TrackSelection = cms.Path(process.L1TrackSelectionProducerExtended)
    process.pL1TrackJets = cms.Path(process.L1TrackJetsExtended)
    process.pL1TrackFastJets = cms.Path(process.L1TrackFastJetsExtended)
    process.pL1GTTInput = cms.Path(process.L1GTTInputProducerExtended)
    process.pL1TrackJetsEmu = cms.Path(process.L1TrackJetsExtendedEmulation)
    process.pTkMET = cms.Path(process.L1TrackerEtMissExtended)
    #process.pTkMETEmu = cms.Path(process.L1TrackerEmuEtMissExtended) #Doesn't exist
    process.pTkMHT = cms.Path(process.L1TrackerHTMissExtended)
    process.pTkMHTEmulator = cms.Path(process.L1TrackerEmuHTMissExtended)
    DISPLACED = 'Displaced'#

# HYBRID: extended tracking
elif (L1TRKALGO == 'HYBRID_PROMPTANDDISP'):
    process.TTTracksEmu = cms.Path(process.L1PromptExtendedHybridTracks)
    process.TTTracksEmuWithTruth = cms.Path(process.L1PromptExtendedHybridTracksWithAssociators)
    process.pL1TrackSelection = cms.Path(process.L1TrackSelectionProducer*process.L1TrackSelectionProducerExtended)
    process.pL1TrackJets = cms.Path(process.L1TrackJets*process.L1TrackJetsExtended)
    process.pL1TrackFastJets = cms.Path(process.L1TrackFastJets*process.L1TrackFastJetsExtended)
    process.pL1GTTInput = cms.Path(process.L1GTTInputProducer*process.L1GTTInputProducerExtended)
    process.pL1TrackJetsEmu = cms.Path(process.L1TrackJetsEmulation*process.L1TrackJetsExtendedEmulation)
    process.pTkMET = cms.Path(process.L1TrackerEtMiss*process.L1TrackerEtMissExtended)
    process.pTkMETEmu = cms.Path(process.L1TrackerEmuEtMiss)
    process.pTkMHT = cms.Path(process.L1TrackerHTMiss*process.L1TrackerHTMissExtended)
    process.pTkMHTEmulator = cms.Path(process.L1TrackerEmuHTMiss*process.L1TrackerEmuHTMissExtended)
    DISPLACED = 'Both'




############################################################
# Define the track ntuple process, MyProcess is the (unsigned) PDGID corresponding to the process which is run
# e.g. single electron/positron = 11
#      single pion+/pion- = 211
#      single muon+/muon- = 13
#      pions in jets = 6
#      taus = 15
#      all TPs = 1
############################################################

process.L1TrackNtuple = cms.EDAnalyzer('L1TrackObjectNtupleMaker',
        MyProcess = cms.int32(1),
        DebugMode = cms.bool(False),      # printout lots of debug statements
        SaveAllTracks = cms.bool(True),  # save *all* L1 tracks, not just truth matched to primary particle
        SaveStubs = cms.bool(False),      # save some info for *all* stubs
        Displaced = cms.string(DISPLACED),# "Prompt", "Displaced", "Both"
        L1Tk_minNStub = cms.int32(4),     # L1 tracks with >= 4 stubs
        TP_minNStub = cms.int32(4),       # require TP to have >= X number of stubs associated with it
        TP_minNStubLayer = cms.int32(4),  # require TP to have stubs in >= X layers/disks
        TP_minPt = cms.double(2.0),       # only save TPs with pt > X GeV
        TP_maxEta = cms.double(2.5),      # only save TPs with |eta| < X
        TP_maxZ0 = cms.double(15.0),      # only save TPs with |z0| < X cm
        L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),                                                      # TTTracks, prompt
        L1TrackExtendedInputTag = cms.InputTag("TTTracksFromExtendedTrackletEmulation", "Level1TTTracks"),                                      # TTTracks, extended
        MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"),                                               # MCTruth track, prompt
        MCTruthTrackExtendedInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigisExtended", "Level1TTTracks"),                               # MCTruth track, extended
        L1TrackGTTInputTag = cms.InputTag("L1GTTInputProducer","Level1TTTracksConverted"),                                                      # TTTracks, prompt, GTT converted
        L1TrackExtendedGTTInputTag = cms.InputTag("L1GTTInputProducerExtended","Level1TTTracksExtendedConverted"),                              # TTTracks, extended, GTT converted
        L1TrackSelectedInputTag = cms.InputTag("L1TrackSelectionProducer", "Level1TTTracksSelected"),                                           # TTTracks, prompt, selected
        L1TrackSelectedEmulationInputTag = cms.InputTag("L1TrackSelectionProducer", "Level1TTTracksSelectedEmulation"),                         # TTTracks, prompt, emulation, selected
        L1TrackExtendedSelectedInputTag = cms.InputTag("L1TrackSelectionProducerExtended", "Level1TTTracksExtendedSelected"),                   # TTTracks, extended, selected
        L1TrackExtendedSelectedEmulationInputTag = cms.InputTag("L1TrackSelectionProducerExtended", "Level1TTTracksExtendedSelectedEmulation"), # TTTracks, extended, emulation, selected
        L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
        MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
        MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
        TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
        TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
        GenJetInputTag = cms.InputTag("ak4GenJets", ""),
        ##track jets and track MET
        SaveTrackJets = cms.bool(True), #includes emulated jets
        SaveTrackSums = cms.bool(True), #includes simulated/emulated track MET, MHT, and HT
        TrackFastJetsInputTag = cms.InputTag("L1TrackFastJets","L1TrackFastJets"),
        TrackFastJetsExtendedInputTag = cms.InputTag("L1TrackFastJetsExtended","L1TrackFastJetsExtended"),
        TrackJetsInputTag = cms.InputTag("L1TrackJets", "L1TrackJets"),
        TrackJetsExtendedInputTag=cms.InputTag("L1TrackJetsExtended", "L1TrackJetsExtended"),
        TrackJetsEmuInputTag = cms.InputTag("L1TrackJetsEmulation","L1TrackJets"),
        TrackJetsExtendedEmuInputTag = cms.InputTag("L1TrackJetsExtendedEmulation","L1TrackJetsExtended"),
        TrackMETInputTag = cms.InputTag("L1TrackerEtMiss","L1TrackerEtMiss"),
        TrackMETExtendedInputTag = cms.InputTag("L1TrackerEtMissExtended","L1TrackerExtendedEtMiss"),
        TrackMETEmuInputTag = cms.InputTag("L1TrackerEmuEtMiss","L1TrackerEmuEtMiss"),
        TrackMHTInputTag = cms.InputTag("L1TrackerHTMiss","L1TrackerHTMiss"), #includes HT
        TrackMHTExtendedInputTag = cms.InputTag("L1TrackerHTMissExtended","L1TrackerHTMissExtended"),
        TrackMHTEmuInputTag = cms.InputTag("L1TrackerEmuHTMiss",process.L1TrackerEmuHTMiss.L1MHTCollectionName.value()),
        TrackMHTEmuExtendedInputTag = cms.InputTag("L1TrackerEmuHTMissExtended",process.L1TrackerEmuHTMissExtended.L1MHTCollectionName.value()),
        GenParticleInputTag = cms.InputTag("genParticles",""),
        RecoVertexInputTag=cms.InputTag("L1VertexFinder", "l1vertices"),
        RecoVertexEmuInputTag=cms.InputTag("L1VertexFinderEmulator", "l1verticesEmulation"),
)

process.ntuple = cms.Path(process.L1TrackNtuple)

process.out = cms.OutputModule( "PoolOutputModule",
                                fastCloning = cms.untracked.bool( False ),
                                fileName = cms.untracked.string("test.root" )
		               )
process.pOut = cms.EndPath(process.out)


# use this if you want to re-run the stub making
# process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubTruth,process.TTTracksEmuWithTruth,process.ntuple)

# use this if cluster/stub associators not available
# process.schedule = cms.Schedule(process.TTClusterStubTruth,process.TTTracksEmuWithTruth,process.ntuple)

process.schedule = cms.Schedule(process.TTClusterStub, process.TTClusterStubTruth, process.dtc, process.TTTracksEmuWithTruth, process.pL1GTTInput, process.pPV, process.pPVemu, process.pL1TrackSelection, process.pL1TrackJets, process.pL1TrackJetsEmu,
process.pL1TrackFastJets, process.pTkMET, process.pTkMETEmu, process.pTkMHT, process.pTkMHTEmulator, process.ntuple)
