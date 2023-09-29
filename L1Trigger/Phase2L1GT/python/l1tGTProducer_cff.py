import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GT.l1tGTScales import scale_parameter

l1tGTProducer = cms.EDProducer(
    "L1GTProducer",
    scales=scale_parameter,
    GTTPromptJets = cms.InputTag("l1tTrackJetsEmulation", "L1TrackJets"),
    GTTDisplacedJets = cms.InputTag("l1tTrackJetsExtendedEmulation", "L1TrackJetsExtended"),
    GTTPrimaryVert = cms.InputTag("l1tVertexFinderEmulator", "l1verticesEmulation"),
    GMTSaPromptMuons = cms.InputTag("l1tSAMuonsGmt", "promptSAMuons"),
    GMTSaDisplacedMuons = cms.InputTag("l1tSAMuonsGmt", "displacedSAMuons"),
    GMTTkMuons = cms.InputTag("l1tTkMuonsGmtLowPtFix", "l1tTkMuonsGmtLowPtFix"),
    CL2Jets = cms.InputTag("l1tSCPFL1PuppiCorrectedEmulator"),
    CL2Electrons = cms.InputTag("l1tLayer2EG", "L1CtTkElectron"),
    CL2Photons = cms.InputTag("l1tLayer2EG", "L1CtTkEm"),
    CL2Taus = cms.InputTag("l1tNNTauProducerPuppi", "L1PFTausNN"),
    CL2EtSum = cms.InputTag("l1tMETPFProducer"),
    CL2HtSum = cms.InputTag("l1tSCPFL1PuppiCorrectedEmulatorMHT")
)
