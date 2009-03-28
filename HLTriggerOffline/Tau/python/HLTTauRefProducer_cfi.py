import FWCore.ParameterSet.Config as cms

TauRefCombiner = cms.EDFilter("HLTTauRefProducer",
    PFTaus = cms.untracked.PSet(
        PFTauDiscriminator = cms.untracked.InputTag("shrinkingConePFTauDiscriminationByIsolation"),
        doPFTaus = cms.untracked.bool(True),
        ptMin = cms.untracked.double(15.0),
        PFTauProducer = cms.untracked.InputTag("fixedConePFTauProducer")
    ),
    MC = cms.untracked.PSet(
        GenParticles = cms.untracked.InputTag("source"),
        ptMinElectron = cms.untracked.double(5.0),
        doMC = cms.untracked.bool(True),
        ptMinTau = cms.untracked.double(10.0),
        BosonID = cms.untracked.int32(23),
        ptMinMuon = cms.untracked.double(3.0)
    ),
    CaloTaus = cms.untracked.PSet(
        ptMinTau = cms.untracked.double(15.0),
        doCaloTaus = cms.untracked.bool(True),
        CaloTauProducer = cms.untracked.InputTag("caloRecoTauProducer"),
        CaloTauDiscriminator = cms.untracked.InputTag("caloRecoTauDiscriminationByIsolation")
    ),
    Electrons = cms.untracked.PSet(
        ElectronCollection = cms.untracked.InputTag("pixelMatchGsfElectrons"),
        doID = cms.untracked.bool(False),
        InnerConeDR = cms.untracked.double(0.02),
        MaxIsoVar = cms.untracked.double(0.02),
        doElectrons = cms.untracked.bool(True),
        TrackCollection = cms.untracked.InputTag("generalTracks"),
        OuterConeDR = cms.untracked.double(0.6),
        ptMin = cms.untracked.double(10.0),
        doTrackIso = cms.untracked.bool(True),
        ptMinTrack = cms.untracked.double(1.5),
        lipMinTrack = cms.untracked.double(0.2),
        IdCollection = cms.untracked.InputTag("elecIDext")
    ),
    Jets = cms.untracked.PSet(
        JetCollection = cms.untracked.InputTag("iterativeCone5CaloJets"),
        etMin = cms.untracked.double(15.0),
        doJets = cms.untracked.bool(True)
    ),
    Muons = cms.untracked.PSet(
        doMuons = cms.untracked.bool(True),
        MuonCollection = cms.untracked.InputTag("muons"),
        ptMin = cms.untracked.double(5.0)
    ),
    EtaMax = cms.untracked.double(2.5)
)



