import FWCore.ParameterSet.Config as cms


BoardDataInput = cms.EDAnalyzer("L1GTObjectBoardWriter",
    GTTPromptJets = cms.InputTag("l1tTrackJetsEmulation", "L1TrackJets"),
    GTTDisplacedJets = cms.InputTag("l1tTrackJetsExtendedEmulation", "L1TrackJetsExtended"),
    GTTPromptHtSum = cms.InputTag("l1tTrackerEmuHTMiss", "L1TrackerEmuHTMiss"),
    GTTDisplacedHtSum = cms.InputTag("l1tTrackerEmuHTMissExtended", "L1TrackerEmuHTMissExtended"),
    GTTEtSum = cms.InputTag("l1tTrackerEmuEtMiss","L1TrackerEmuEtMiss"),
    GTTPrimaryVert = cms.InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"),
    GMTSaPromptMuons = cms.InputTag("l1tSAMuonsGmt", "promptSAMuons"),
    GMTSaDisplacedMuons = cms.InputTag("l1tSAMuonsGmt", "displacedSAMuons"),
    GMTTkMuons = cms.InputTag("l1tTkMuonsGmtLowPtFix", "l1tTkMuonsGmtLowPtFix"),
    CL2JetsSC4 = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulator"),
    CL2JetsSC8 = cms.InputTag("l1tSC8PFL1PuppiCorrectedEmulator"),
    CL2Electrons = cms.InputTag("l1tLayer2EG", "L1CtTkElectron"),
    CL2Photons = cms.InputTag("l1tLayer2EG", "L1CtTkEm"),
    CL2Taus = cms.InputTag("l1tNNTauProducerPuppi", "L1PFTausNN"),
    CL2EtSum = cms.InputTag("l1tMETPFProducer"),
    CL2HtSum = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulatorMHT"),
    filename = cms.string("inputPattern"),
    maxEvents = cms.uint32(72),
    bufferFileType = cms.string("input")
)

BoardDataOutputObjects = cms.EDAnalyzer("L1GTObjectBoardWriter",
    GTTPromptJets = cms.InputTag("l1tTrackJetsEmulation", "L1TrackJets"),
    GTTDisplacedJets = cms.InputTag("l1tTrackJetsExtendedEmulation", "L1TrackJetsExtended"),
    GTTPromptHtSum = cms.InputTag("l1tTrackerEmuHTMiss", "L1TrackerEmuHTMiss"),
    GTTDisplacedHtSum = cms.InputTag("l1tTrackerEmuHTMissExtended", "L1TrackerEmuHTMissExtended"),
    GTTEtSum = cms.InputTag("l1tTrackerEmuEtMiss","L1TrackerEmuEtMiss"),
    GTTPrimaryVert = cms.InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"),
    GMTSaPromptMuons = cms.InputTag("l1tSAMuonsGmt", "promptSAMuons"),
    GMTSaDisplacedMuons = cms.InputTag("l1tSAMuonsGmt", "displacedSAMuons"),
    GMTTkMuons = cms.InputTag("l1tTkMuonsGmtLowPtFix", "l1tTkMuonsGmtLowPtFix"),
    CL2JetsSC4 = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulator"),
    CL2JetsSC8 = cms.InputTag("l1tSC8PFL1PuppiCorrectedEmulator"),
    CL2Electrons = cms.InputTag("l1tLayer2EG", "L1CtTkElectron"),
    CL2Photons = cms.InputTag("l1tLayer2EG", "L1CtTkEm"),
    CL2Taus = cms.InputTag("l1tNNTauProducerPuppi", "L1PFTausNN"),
    CL2EtSum = cms.InputTag("l1tMETPFProducer"),
    CL2HtSum = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulatorMHT"),
    filename = cms.string("outputObjectPattern"),
    maxEvents = cms.uint32(72),
    bufferFileType = cms.string("output")
)

AlgoBitBoardData = cms.EDAnalyzer("L1GTAlgoBoardWriter",
    outputFilename = cms.string("algoBitPattern"),
    algoBlocksTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    maxEvents = cms.uint32(72)
)
