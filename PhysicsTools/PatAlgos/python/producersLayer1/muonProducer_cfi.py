import FWCore.ParameterSet.Config as cms

allLayer1Muons = cms.EDProducer("PATMuonProducer",
    addGenMatch = cms.bool(True),
    addResolutions = cms.bool(True),
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
    addLRValues = cms.bool(True),
    isoDeposits = cms.PSet(
        hcal = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
        tracker = cms.InputTag("layer0MuonIsolations","muIsoDepositTk"),
        user = cms.VInputTag(cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersho"), cms.InputTag("layer0MuonIsolations","muIsoDepositJets")),
        ecal = cms.InputTag("layer0MuonIsolations","muIsoDepositCalByAssociatorTowersecal")
    ),
    tracksSource = cms.InputTag("generalTracks"),
    useNNResolutions = cms.bool(False),
    addMuonID = cms.bool(True),
    muonLRFile = cms.string('PhysicsTools/PatUtils/data/MuonLRDistros.root'),
    muonSource = cms.InputTag("allLayer0Muons"),
    muonResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_muon.root'),
    genParticleMatch = cms.InputTag("muonMatch")
)


