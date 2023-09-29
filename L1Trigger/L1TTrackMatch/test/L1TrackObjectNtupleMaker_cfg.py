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
process.load('Configuration.Geometry.GeometryExtended2026D95Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D95_cff')
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
                                  '/store/relval/CMSSW_13_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/130X_mcRun4_realistic_v2_2026D95noPU-v1/00000/16f6615d-f98c-475f-ad33-0e89934b6c7f.root'
)
secFiles = cms.untracked.vstring()

process.source = cms.Source ("PoolSource",
                            fileNames = readFiles,
                            secondaryFileNames = secFiles,
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            )

process.source.inputCommands = cms.untracked.vstring("keep *","drop l1tTkPrimaryVertexs_L1TkPrimaryVertex__*")
process.Timing = cms.Service("Timing",
  summaryOnly = cms.untracked.bool(False),
  useJobReport = cms.untracked.bool(True)
)

process.TFileService = cms.Service("TFileService", fileName = cms.string('GTTObjects_ttbar200PU_v2p2.root'), closeFileFast = cms.untracked.bool(True))


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
process.load('L1Trigger.TrackerDTC.ProducerED_cff')
process.dtc = cms.Path(process.TrackerDTCProducer)

process.load("L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff")
process.load("L1Trigger.L1TTrackMatch.l1tTrackSelectionProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackVertexAssociationProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackJets_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tGTTInputProducer_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackJetsEmulation_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackFastJets_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackerEtMiss_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackerEmuEtMiss_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackerHTMiss_cfi")
process.load("L1Trigger.L1TTrackMatch.l1tTrackerEmuHTMiss_cfi")
process.load('L1Trigger.VertexFinder.l1tVertexProducer_cfi')


############################################################
# Primary vertex
############################################################
process.pPV = cms.Path(process.l1tVertexFinder)
process.pPVemu = cms.Path(process.l1tVertexFinderEmulator)

# HYBRID: prompt tracking
if (L1TRKALGO == 'HYBRID'):
    process.TTTracksEmu = cms.Path(process.L1THybridTracks)
    process.TTTracksEmuWithTruth = cms.Path(process.L1THybridTracksWithAssociators)
    process.pL1TrackSelection = cms.Path(process.l1tTrackSelectionProducer *
                                         process.l1tTrackSelectionProducerForJets *
                                         process.l1tTrackSelectionProducerForEtMiss)
    process.pL1TrackVertexAssociation = cms.Path(process.l1tTrackVertexAssociationProducer*
                                                 process.l1tTrackVertexAssociationProducerForJets*
                                                 process.l1tTrackVertexAssociationProducerForEtMiss)
    process.pL1TrackJets = cms.Path(process.l1tTrackJets)
    process.pL1TrackFastJets=cms.Path(process.l1tTrackFastJets)
    process.pL1GTTInput = cms.Path(process.l1tGTTInputProducer)
    process.pL1TrackJetsEmu = cms.Path(process.l1tTrackJetsEmulation)
    process.pTkMET = cms.Path(process.l1tTrackerEtMiss)
    process.pTkMETEmu = cms.Path(process.l1tTrackerEmuEtMiss)
    process.pTkMHT = cms.Path(process.l1tTrackerHTMiss)
    process.pTkMHTEmulator = cms.Path(process.l1tTrackerEmuHTMiss)
    DISPLACED = 'Prompt'

# HYBRID: extended tracking
elif (L1TRKALGO == 'HYBRID_DISPLACED'):
    process.TTTracksEmu = cms.Path(process.L1TExtendedHybridTracks)
    process.TTTracksEmuWithTruth = cms.Path(process.L1TExtendedHybridTracksWithAssociators)
    process.pL1TrackSelection = cms.Path(process.l1tTrackSelectionProducer *
                                         process.l1tTrackSelectionProducerExtended *
                                         process.l1tTrackSelectionProducerExtendedForJets *
                                         process.l1tTrackSelectionProducerExtendedForEtMiss)
    process.pL1TrackVertexAssociation = cms.Path(process.l1tTrackVertexAssociationProducerExtended *
                                                 process.l1tTrackVertexAssociationProducerExtendedForJets *
                                                 process.l1tTrackVertexAssociationProducerExtendedForEtMiss)
    process.pL1TrackJets = cms.Path(process.l1tTrackJetsExtended)
    process.pL1TrackFastJets = cms.Path(process.l1tTrackFastJetsExtended)
    process.pL1GTTInput = cms.Path(process.l1tGTTInputProducerExtended)
    process.pL1TrackJetsEmu = cms.Path(process.l1tTrackJetsExtendedEmulation)
    process.pTkMET = cms.Path(process.l1tTrackerEtMissExtended)
    process.pTkMHT = cms.Path(process.l1tTrackerHTMissExtended)
    process.pTkMHTEmulator = cms.Path(process.l1tTrackerEmuHTMissExtended)
    DISPLACED = 'Displaced'#

# HYBRID: extended tracking
elif (L1TRKALGO == 'HYBRID_PROMPTANDDISP'):
    process.TTTracksEmu = cms.Path(process.L1TPromptExtendedHybridTracks)
    process.TTTracksEmuWithTruth = cms.Path(process.L1TPromptExtendedHybridTracksWithAssociators)
    process.pL1TrackSelection = cms.Path(process.l1tTrackSelectionProducer * process.l1tTrackSelectionProducerExtended *
                                         process.l1tTrackSelectionProducerForJets * process.l1tTrackSelectionProducerExtendedForJets *
                                         process.l1tTrackSelectionProducerForEtMiss * process.l1tTrackSelectionProducerExtendedForEtMiss)
    process.pL1TrackVertexAssociation = cms.Path(process.l1tTrackVertexAssociationProducer * process.l1tTrackVertexAssociationProducerExtended *
                                                 process.l1tTrackVertexAssociationProducerForJets * process.l1tTrackVertexAssociationProducerExtendedForJets *
                                                 process.l1tTrackVertexAssociationProducerForEtMiss * process.l1tTrackVertexAssociationProducerExtendedForEtMiss)
    process.pL1TrackJets = cms.Path(process.l1tTrackJets*process.l1tTrackJetsExtended)
    process.pL1TrackFastJets = cms.Path(process.l1tTrackFastJets*process.l1tTrackFastJetsExtended)
    process.pL1GTTInput = cms.Path(process.l1tGTTInputProducer*process.l1tGTTInputProducerExtended)
    process.pL1TrackJetsEmu = cms.Path(process.l1tTrackJetsEmulation*process.l1tTrackJetsExtendedEmulation)
    process.pTkMET = cms.Path(process.l1tTrackerEtMiss*process.l1tTrackerEtMissExtended)
    process.pTkMETEmu = cms.Path(process.l1tTrackerEmuEtMiss)
    process.pTkMHT = cms.Path(process.l1tTrackerHTMiss*process.l1tTrackerHTMissExtended)
    process.pTkMHTEmulator = cms.Path(process.l1tTrackerEmuHTMiss*process.l1tTrackerEmuHTMissExtended)
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
        L1TrackInputTag = cms.InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"),                                                      # TTTracks, prompt
        L1TrackExtendedInputTag = cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation", "Level1TTTracks"),                                      # TTTracks, extended
        MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"),                                               # MCTruth track, prompt
        MCTruthTrackExtendedInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigisExtended", "Level1TTTracks"),                               # MCTruth track, extended
        L1TrackGTTInputTag = cms.InputTag("l1tGTTInputProducer","Level1TTTracksConverted"),                                                      # TTTracks, prompt, GTT converted
        L1TrackExtendedGTTInputTag = cms.InputTag("l1tGTTInputProducerExtended","Level1TTTracksExtendedConverted"),                              # TTTracks, extended, GTT converted
        L1TrackSelectedInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelected"),                                           # TTTracks, prompt, selected
        L1TrackSelectedEmulationInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelectedEmulation"),                         # TTTracks, prompt, emulation, selected
        L1TrackSelectedAssociatedInputTag = cms.InputTag("l1tTrackVertexAssociationProducer", "Level1TTTracksSelectedAssociated"),                                           # TTTracks, prompt, selected, associated
        L1TrackSelectedAssociatedEmulationInputTag = cms.InputTag("l1tTrackVertexAssociationProducer", "Level1TTTracksSelectedAssociatedEmulation"),                         # TTTracks, prompt, emulation, selected, associated
                                       
        L1TrackSelectedForJetsInputTag = cms.InputTag("l1tTrackSelectionProducerForJets", "Level1TTTracksSelected"),                                           # TTTracks, prompt, selected
        L1TrackSelectedEmulationForJetsInputTag = cms.InputTag("l1tTrackSelectionProducerForJets", "Level1TTTracksSelectedEmulation"),                         # TTTracks, prompt, emulation, selected
        L1TrackSelectedAssociatedForJetsInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForJets", "Level1TTTracksSelectedAssociated"),                                           # TTTracks, prompt, selected, associated
        L1TrackSelectedAssociatedEmulationForJetsInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForJets", "Level1TTTracksSelectedAssociatedEmulation"),                         # TTTracks, prompt, emulation, selected, associated
                                       
        L1TrackSelectedForEtMissInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss", "Level1TTTracksSelected"),                                           # TTTracks, prompt, selected
        L1TrackSelectedEmulationForEtMissInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss", "Level1TTTracksSelectedEmulation"),                         # TTTracks, prompt, emulation, selected
        L1TrackSelectedAssociatedForEtMissInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForEtMiss", "Level1TTTracksSelectedAssociated"),                                           # TTTracks, prompt, selected, associated
        L1TrackSelectedAssociatedEmulationForEtMissInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForEtMiss", "Level1TTTracksSelectedAssociatedEmulation"),                         # TTTracks, prompt, emulation, selected, associated

        L1TrackExtendedSelectedInputTag = cms.InputTag("l1tTrackSelectionProducerExtended", "Level1TTTracksExtendedSelected"),                                           # TTTracks, extended, selected
        L1TrackExtendedSelectedEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerExtended", "Level1TTTracksExtendedSelectedEmulation"),                         # TTTracks, extended, emulation, selected
        L1TrackExtendedSelectedAssociatedInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtended", "Level1TTTracksExtendedSelectedAssociated"),                                           # TTTracks, extended, selected, associated
        L1TrackExtendedSelectedAssociatedEmulationInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtended", "Level1TTTracksExtendedSelectedAssociatedEmulation"),                         # TTTracks, extended, emulation, selected, associated
                                       
        L1TrackExtendedSelectedForJetsInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForJets", "Level1TTTracksExtendedSelected"),                                           # TTTracks, extended, selected
        L1TrackExtendedSelectedEmulationForJetsInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForJets", "Level1TTTracksExtendedSelectedEmulation"),                         # TTTracks, extended, emulation, selected
        L1TrackExtendedSelectedAssociatedForJetsInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtendedForJets", "Level1TTTracksExtendedSelectedAssociated"),                                           # TTTracks, extended, selected, associated
        L1TrackExtendedSelectedAssociatedEmulationForJetsInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtendedForJets", "Level1TTTracksExtendedSelectedAssociatedEmulation"),                         # TTTracks, extended, emulation, selected, associated
                                       
        L1TrackExtendedSelectedForEtMissInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForEtMiss", "Level1TTTracksExtendedSelected"),                                           # TTTracks, extended, selected
        L1TrackExtendedSelectedEmulationForEtMissInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForEtMiss", "Level1TTTracksExtendedSelectedEmulation"),                         # TTTracks, extended, emulation, selected
        L1TrackExtendedSelectedAssociatedForEtMissInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtendedForEtMiss", "Level1TTTracksExtendedSelectedAssociated"),                                           # TTTracks, extended, selected, associated
        L1TrackExtendedSelectedAssociatedEmulationForEtMissInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtendedForEtMiss", "Level1TTTracksExtendedSelectedAssociatedEmulation"),                         # TTTracks, extended, emulation, selected, associated
                                                                              
        L1StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"),
        MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
        MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
        TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
        TrackingVertexInputTag = cms.InputTag("mix", "MergedTrackTruth"),
        GenJetInputTag = cms.InputTag("ak4GenJets", ""),
        ##track jets and track MET
        SaveTrackJets = cms.bool(True), #includes emulated jets
        SaveTrackSums = cms.bool(True), #includes simulated/emulated track MET, MHT, and HT
        TrackFastJetsInputTag = cms.InputTag("l1tTrackFastJets","L1TrackFastJets"),
        TrackFastJetsExtendedInputTag = cms.InputTag("l1tTrackFastJetsExtended","L1TrackFastJetsExtended"),
        TrackJetsInputTag = cms.InputTag("l1tTrackJets", "L1TrackJets"),
        TrackJetsExtendedInputTag=cms.InputTag("l1tTrackJetsExtended", "L1TrackJetsExtended"),
        TrackJetsEmuInputTag = cms.InputTag("l1tTrackJetsEmulation","L1TrackJets"),
        TrackJetsExtendedEmuInputTag = cms.InputTag("l1tTrackJetsExtendedEmulation","L1TrackJetsExtended"),
        TrackMETInputTag = cms.InputTag("l1tTrackerEtMiss","L1TrackerEtMiss"),
        TrackMETExtendedInputTag = cms.InputTag("l1tTrackerEtMissExtended","L1TrackerExtendedEtMiss"),
        TrackMETEmuInputTag = cms.InputTag("l1tTrackerEmuEtMiss","L1TrackerEmuEtMiss"),
        TrackMHTInputTag = cms.InputTag("l1tTrackerHTMiss","L1TrackerHTMiss"), #includes HT
        TrackMHTExtendedInputTag = cms.InputTag("l1tTrackerHTMissExtended","L1TrackerHTMissExtended"),
        TrackMHTEmuInputTag = cms.InputTag("l1tTrackerEmuHTMiss",process.l1tTrackerEmuHTMiss.L1MHTCollectionName.value()),
        TrackMHTEmuExtendedInputTag = cms.InputTag("l1tTrackerEmuHTMissExtended",process.l1tTrackerEmuHTMissExtended.L1MHTCollectionName.value()),
        GenParticleInputTag = cms.InputTag("genParticles",""),
        RecoVertexInputTag=cms.InputTag("l1tVertexFinder", "L1Vertices"),
        RecoVertexEmuInputTag=cms.InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"),
)

process.ntuple = cms.Path(process.L1TrackNtuple)

process.out = cms.OutputModule( "PoolOutputModule",
                                outputCommands = process.RAWSIMEventContent.outputCommands,
                                fileName = cms.untracked.string("test.root" )
		               )
process.out.outputCommands.append('keep  *TTTrack*_*_*_*')
process.pOut = cms.EndPath(process.out)


# use this if you want to re-run the stub making
# process.schedule = cms.Schedule(process.TTClusterStub,process.TTClusterStubTruth,process.TTTracksEmuWithTruth,process.ntuple)

# use this if cluster/stub associators not available
# process.schedule = cms.Schedule(process.TTClusterStubTruth,process.TTTracksEmuWithTruth,process.ntuple)

process.schedule = cms.Schedule(process.TTClusterStub, process.TTClusterStubTruth, process.dtc, process.TTTracksEmuWithTruth, process.pL1GTTInput, process.pL1TrackSelection, process.pPV, process.pPVemu,process.pL1TrackVertexAssociation, process.pL1TrackJets, process.pL1TrackJetsEmu,process.pL1TrackFastJets, process.pTkMET, process.pTkMETEmu, process.pTkMHT, process.pTkMHTEmulator, process.ntuple)
