import FWCore.ParameterSet.Config as cms

l1tau = cms.EDAnalyzer("L1TauAnalyzer",
    BosonPID = cms.int32(23),
    # L1extra reading
    L1extraTauJetSource = cms.InputTag("l1extraParticles","Tau"),
    DoPFTauMatching = cms.bool(True),
    PFTauMinEt = cms.double(15.0),
    L1extraNonIsoEgammaSource = cms.InputTag("l1extraParticles","NonIsolated"),
    PFTauDiscriminatorSource = cms.InputTag("pfRecoTauDiscriminationByIsolation"),
    PFTauSource = cms.InputTag("pfRecoTauProducer"),
    L1extraIsoEgammaSource = cms.InputTag("l1extraParticles","Isolated"),
    SingleTauMETThresholds = cms.vdouble(30.0, 30.0),
    L1extraMETSource = cms.InputTag("l1extraParticles"),
    L1extraCenJetSource = cms.InputTag("l1extraParticles","Central"),
    PFMCTauMinDeltaR = cms.double(0.15),
    SingleTauThreshold = cms.double(80.0),
    MCTauHadMinEt = cms.double(15.0),
    GenParticleSource = cms.InputTag("source"),
    L1IsoEGTauName = cms.string('L1_IsoEG10_TauJet20'),
    L1GtObjectMap = cms.InputTag("l1GtEmulDigis"),
    MuTauThresholds = cms.vdouble(5.0, 20.0),
    # GT bit reading
    L1GtReadoutRecord = cms.InputTag("l1GtEmulDigis"),
    L1SingleTauName = cms.string('L1_SingleTauJet80'),
    PFTauMaxAbsEta = cms.double(2.5),
    L1DoubleTauName = cms.string('L1_DoubleTauJet40'),
    DoMCMatching = cms.bool(True),
    L1TauMETName = cms.string('L1_TauJet30_ETM30'),
    MCTauHadMaxAbsEta = cms.double(2.5),
    IsoEgTauThresholds = cms.vdouble(10.0, 20.0),
    L1MuonTauName = cms.string('L1_Mu5_TauJet20'),
    # int32    BosonPID     = 37 //(H+)
    # int32    BosonPID     = 35 //(H0)
    # int32    BosonPID     = 36 //(A0)
    L1MCTauMinDeltaR = cms.double(0.5),
    DoubleTauThreshold = cms.double(40.0),
    L1extraForJetSource = cms.InputTag("l1extraParticles","Forward"),
    L1extraMuonSource = cms.InputTag("l1extraParticles")
)


