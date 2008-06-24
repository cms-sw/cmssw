import FWCore.ParameterSet.Config as cms

allLayer0PFJets = cms.EDFilter("PATPFJetCleaner",
    ## uncalibrated reco jet input source
    jetSource = cms.InputTag("iterativeCone5PFJets"), 

    # selection (e.g. ID)
    selection = cms.PSet(
        type = cms.string('none')
    ),

    removeOverlaps = cms.PSet(
        ## Flag or discard jets that match with clean electrons (should be done better on PFlow Jets!)
        electrons = cms.PSet( 
            collection = cms.InputTag("allLayer0Electrons"), ## 
            deltaR = cms.double(0.3), ##
            cut = cms.string('pt > 10'),              ## as in LeptonJetIsolationAngle
            flags = cms.vstring('Isolation/Tracker'), ## request the item to be marked as isolated in the tracker
                                                      ## by the PATElectronCleaner
        ),
        ## Flag or discard jets that match with Taus. Off, as it was not there in TQAF
        #taus = cms.PSet(     
        #    collection = cms.InputTag("allLayer0Taus")
        #    deltaR     = cms.double(0.3)
        #)
        ## flag or discard jets that match with Photons. Off, as it was not there in TQAF
        #photons = cms.PSet(
        #    collection = cms.InputTag("allLayer0Photons")
        #    deltaR     = cms.double(0.3)
        #)
        #muons = cms.PSet( ... ) // supported, but it's not likely you want it
        #jets  = cms.PSet( ... ) // same as above   
        user = cms.VPSet()
    ),

    markItems = cms.bool(True),    ## write the status flags in the output items
    bitsToIgnore = cms.vstring('Overlap/All'), ## You can specify some bit names, e.g. "Overflow/User1", "Core/Duplicate", "Isolation/All".
    saveRejected = cms.string(''), ## set this to a non empty label to save the list of items which fail
    saveAll = cms.string(''),      ## set this to a non empty label to save a list of all items both passing and failing
)


