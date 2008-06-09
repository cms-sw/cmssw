import FWCore.ParameterSet.Config as cms

allLayer0Muons = cms.EDFilter("PATMuonCleaner",
    selection = cms.PSet(
        flag = cms.string('TMLastStationLoose'),
        type = cms.string('muonPOG'),
        minCaloCompatibility = cms.double(0.0),
        minSegmentCompatibility = cms.double(0.0)
    ),
    muonSource = cms.InputTag("muons"),
    saveRejected = cms.string(''),
    bitsToIgnore = cms.vstring('Isolation/All'),
    wantSummary = cms.untracked.bool(True),
    markItems = cms.bool(True),
    saveAll = cms.string(''),
    isolation = cms.PSet(
        hcal = cms.PSet(
            src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowersecal"),
            deltaR = cms.double(0.3),
            cut = cms.double(2.0)
        ),
        tracker = cms.PSet(
            src = cms.InputTag("patAODMuonIsolations","muIsoDepositTk"),
            deltaR = cms.double(0.3),
            cut = cms.double(2.0)
        ),
        user = cms.VPSet(cms.PSet(
            src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowersho"),
            deltaR = cms.double(0.3),
            cut = cms.double(2.0)
        ), 
            cms.PSet(
                src = cms.InputTag("patAODMuonIsolations","muIsoDepositJets"),
                deltaR = cms.double(0.5),
                cut = cms.double(2.0)
            )),
        ecal = cms.PSet(
            src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
            deltaR = cms.double(0.3),
            cut = cms.double(2.0)
        )
    )
)


