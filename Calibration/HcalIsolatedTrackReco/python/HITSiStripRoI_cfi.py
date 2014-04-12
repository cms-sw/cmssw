import FWCore.ParameterSet.Config as cms

HITSiStripRoI = cms.EDProducer("HITSiStripRawToClustersRoI",
    # layers of interest
    layers = cms.untracked.int32(10),
    ptrackEtaWindow = cms.untracked.double(0.3),
    pixelTrackLabel = cms.InputTag("isolPixelTrackFilterL2"),
    # define objects of interest
    doGlobal = cms.untracked.bool(False),
    usePixelTracks = cms.untracked.bool(True),
    l1tauJetLabel = cms.InputTag("l1extraParticles","Tau"),
    random = cms.untracked.bool(False),
    tjetPhiWindow = cms.untracked.double(0.05),
    # define tracker windows
    tjetEtaWindow = cms.untracked.double(0.05),
    # define input tags
    siStripLazyGetter = cms.InputTag("siStripRawToClustersFacilityHIT"),
    useTauJets = cms.untracked.bool(False),
    ptrackPhiWindow = cms.untracked.double(0.3)
)


