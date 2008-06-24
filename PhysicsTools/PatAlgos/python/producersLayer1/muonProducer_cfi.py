import FWCore.ParameterSet.Config as cms

allLayer1Muons = cms.EDProducer("PATMuonProducer",
    # General configurables
    muonSource = cms.InputTag("allLayer0Muons"),

    embedTrack          = cms.bool(False), ## whether to embed in AOD externally stored tracker track
    embedCombinedMuon   = cms.bool(False), ## whether to embed in AOD externally stored combined muon track
    embedStandAloneMuon = cms.bool(False), ## whether to embed in AOD externally stored standalone muon track

    # isolation configurables
    isolation = cms.PSet(
        hcal = cms.PSet(
            src = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
            deltaR = cms.double(0.3)
        ),
        tracker = cms.PSet(
            src = cms.InputTag("layer0MuonIsolations","muIsoDepositTk"),
            deltaR = cms.double(0.3)
        ),
        user = cms.VPSet(cms.PSet(
            src = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersho"),
            deltaR = cms.double(0.3)
        ), 
            cms.PSet(
                src = cms.InputTag("layer0MuonIsolations","muIsoDepositJets"),
                deltaR = cms.double(0.3)
            )),
        ecal = cms.PSet(
            src = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersecal"),
            deltaR = cms.double(0.3)
        )
    ),
    # embed IsoDeposits to recompute isolation easily
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("layer0MuonIsolations","muIsoDepositTk"),
        ecal    = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersecal"),
        hcal    = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
        user    = cms.VInputTag(
                     cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersho"), 
                     cms.InputTag("layer0MuonIsolations","muIsoDepositJets")
                  ),
    ),

    # Muon ID configurables
    addMuonID = cms.bool(False), ## DEPRECATED OLD TQAF muon ID. 

    # Resolution configurables
    addResolutions = cms.bool(True),
    muonResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_muon.root'),
    useNNResolutions = cms.bool(False), ## use the neural network approach?

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    trigPrimMatch = cms.VInputTag(cms.InputTag("muonTrigMatchHLT1MuonNonIso"), cms.InputTag("muonTrigMatchHLT1MET65")),

    # MC matching configurables
    addGenMatch = cms.bool(True),
    genParticleMatch = cms.InputTag("muonMatch") ## particles source to be used for the matching

)


