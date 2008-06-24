import FWCore.ParameterSet.Config as cms

allLayer0TrackCands = cms.EDFilter("PATGenericParticleCleaner",
    ## Input collection (anything readable with View<Candidate>
    src = cms.InputTag("patAODTrackCands"),

    isolation = cms.PSet(
        tracker = cms.PSet(
            src = cms.InputTag("patAODTrackIsolations","patAODTrackIsoDepositCtfTk"), # IsoDeposit
            deltaR    = cms.double(0.3),   # outer cone radius
            veto      = cms.double(0.015), # inner cone radius
            threshold = cms.double(1.5),   # threshold on track Pt
            skipDefaultVeto = cms.bool(True), # ignore default veto in IsoDeposit 
            cut = cms.double(3.0),  
        ),
        ecal = cms.PSet(
            src = cms.InputTag("patAODTrackIsolations","patAODTrackIsoDepositCalByAssociatorTowersecal"), # IsoDeposit
            deltaR = cms.double(0.3),
            cut = cms.double(3.0)
        ),
        hcal = cms.PSet(
            src = cms.InputTag("patAODTrackIsolations","patAODTrackIsoDepositCalByAssociatorTowershcal"), # IsoDeposit
            deltaR = cms.double(0.3), 
            cut = cms.double(5.0)
        ),
    ),

    removeOverlaps = cms.PSet(
        muons = cms.PSet(
            collection = cms.InputTag("allLayer0Muons"),
            deltaR = cms.double(0.3),
            checkRecoComponents = cms.bool(True), # Require them to 'overlap' according to the Candidate OverlapChecker tool
                                                  # that is, to have the same track / globalMuon track / standaloneMuon track
        ),
        electrons = cms.PSet(
            collection = cms.InputTag("allLayer0Electrons"),
            deltaR = cms.double(0.3),
            checkRecoComponents = cms.bool(False), # GsfTrack is never 'equal' to a normal Track, so this must be FALSE
        ),
        taus = cms.PSet(
            collection = cms.InputTag("allLayer0Taus"),
            deltaR = cms.double(0.3),
        ),
    ),

    markItems    = cms.bool(True), ## write the status flags in the output items
    bitsToIgnore = cms.vstring('Isolation/All',  ## Keep but flag non isolated
                               'Overlap/All'),   ## Keep but flag those overlapping with other items
    saveRejected = cms.string(''), ## set this to a non empty label to save the list of items which fail
    saveAll      = cms.string(''), ## set this to a non empty label to save a list of all items both passing and failing

)


