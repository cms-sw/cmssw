import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTScales import scale_parameter

l1tGTProducer = cms.EDProducer(
    "L1GTProducer",
    scales=scale_parameter,
    GTTPromptJets = cms.InputTag("l1tTrackJetsEmulation", "L1TrackJets"),
    GTTDisplacedJets = cms.InputTag("l1tTrackJetsExtendedEmulation", "L1TrackJetsExtended"),
    GTTPrimaryVert = cms.InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"),
    GMTSaPromptMuons = cms.InputTag("l1tSAMuonsGmt", "prompt"),
    GMTSaDisplacedMuons = cms.InputTag("l1tSAMuonsGmt", "displaced"),
    GMTTkMuons = cms.InputTag("l1tTkMuonsGmt"),
    CL2JetsSC4 = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulator"),
    CL2JetsSC8 = cms.InputTag("l1tSC8PFL1PuppiCorrectedEmulator"),
    CL2Electrons = cms.InputTag("l1tLayer2EG", "L1CtTkElectron"),
    CL2Photons = cms.InputTag("l1tLayer2EG", "L1CtTkEm"),
    CL2Taus = cms.InputTag("l1tNNTauProducerPuppi", "L1PFTausNN"),
    CL2EtSum = cms.InputTag("l1tMETPFProducer"),
    CL2HtSum = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulatorMHT")
)
