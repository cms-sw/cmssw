# The following comments couldn't be translated into the new config version:

# set isolation but don't reject non-isolated electrons
import FWCore.ParameterSet.Config as cms

allLayer0Muons = cms.EDFilter("PATMuonCleaner",
    ## reco muon input source
    muonSource = cms.InputTag("muons"), 

    # selection (e.g. ID)
    selection = cms.PSet(
        type = cms.string('none')
    ),
    # Other possible selections:
    ## Reco-based muon selection)
    # selection = cms.PSet( type = cms.string("globalMuons") ) # pick only globalMuons
    ## ID-based selection (maybe only tracker muons?)
    # selection = cms.PSet(                                   
    #     type = cms.string("muonPOG")
    #     flag = cms.string("TMLastStationLoose")     # flag for the muon id algorithm
    #                      # "TMLastStationLoose", "TMLastStationTight"  
    #                      # "TM2DCompatibilityLoose", "TM2DCompatibilityTight" 
    #     minCaloCompatibility    = cms.double(0.0)     # cut on calo compatibility
    #     minSegmentCompatibility = cms.double(0.0)     # cut on muon segment match to tracker
    # )
    ## Custom cut-based selection (from SusyAnalyzer))
    # selection = cms.PSet( type = cms.string("custom")  
    #                       dPbyPmax = cms.double(0.5)
    #                       chi2max  = cms.double(3.0)
    #                       nHitsMin = cms.double(13) )


    # isolation (not mandatory, see bitsToIgnore below)
    isolation = cms.PSet(
        tracker = cms.PSet(
            src = cms.InputTag("patAODMuonIsolations","muIsoDepositTk"),
            deltaR = cms.double(0.3),
            cut = cms.double(2.0) # just a test value, not optimized nor 'official'
        ),
        ecal = cms.PSet(
            src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowersecal"),
            deltaR = cms.double(0.3),
            cut = cms.double(2.0) # just a test value, not optimized nor 'official'
        ),
        hcal = cms.PSet(
            src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
            deltaR = cms.double(0.3),
            cut = cms.double(2.0), # just a test value, not optimized nor 'official'
        ),
        user = cms.VPSet(cms.PSet(
                src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowersho"),
                deltaR = cms.double(0.3),
                cut = cms.double(2.0)     # just a test value, not optimized nor 'official'
            ),cms.PSet(
                src = cms.InputTag("patAODMuonIsolations","muIsoDepositJets"),
                deltaR = cms.double(0.5),
                cut = cms.double(2.0)     # just a test value, not optimized nor 'official'
            )
        ),
    ),

    markItems    = cms.bool(True), ## write the status flags in the output items
    bitsToIgnore = cms.vstring('Isolation/All'), ## Keep non isolated muons, but flag them
                                   ## You can specify some bit names, e.g. "Overflow/User1", "Core/Duplicate", "Isolation/All".
    saveAll      = cms.string(''), ## set this to a non empty label to save a list of all items both passing and failing
    saveRejected = cms.string(''), ## set this to a non empty label to save the list of items which fail
)


