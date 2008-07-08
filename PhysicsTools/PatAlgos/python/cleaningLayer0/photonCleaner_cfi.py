import FWCore.ParameterSet.Config as cms

allLayer0Photons = cms.EDFilter("PATPhotonCleaner",
    ## Input collection of Photons
    photonSource = cms.InputTag("photons"),

    ## Remove internal duplicates (photons with the same superCluster seed or the same superCluster)
    removeDuplicates = cms.string('bySeed'), # 'bySeed', 'bySuperCluster' or none

    ## Remove or flag photons that overlap with electrons (same superCluster seed or same superCluster)
    removeElectrons = cms.string('bySeed'),  # 'bySeed', 'bySuperCluster' or none
    electrons       = cms.InputTag("allLayer0Electrons"),

    isolation = cms.PSet(
        tracker = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("patAODPhotonIsolations","gamIsoDepositTk"),
            # parameters (E/gamma POG defaults)
            deltaR = cms.double(0.3),              # Cone radius
            vetos  = cms.vstring('0.015',          # Inner veto cone radius
                                'Threshold(1.0)'), # Pt threshold
            skipDefaultVeto = cms.bool(True),
            # cut value - just a test, not an official one
            cut = cms.double(3.0),
        ),
        ecal = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("patAODPhotonIsolations","gamIsoDepositEcalFromHits"), # FromClusts if computed from AOD
            # parameters (E/gamma POG defaults)
            deltaR          = cms.double(0.4),
            vetos           = cms.vstring('EcalBarrel:0.045', 'EcalEndcaps:0.070'),
            skipDefaultVeto = cms.bool(True),
            # cut value - just a test, not an official one
            cut = cms.double(5.0),
        ),
        ## other option, using gamIsoDepositEcalSCVetoFromClust (see also recoLayer0/photonIsolation_cff.py)
        #PSet ecal = cms.PSet( 
        #   src    = cms.InputTag("patAODPhotonIsolations", "gamIsoDepositEcalSCVetoFromClusts")
        #   deltaR = cms.double(0.4)
        #   vetos  = cms.vstring()     # no veto, already done with SC
        #   skipDefaultVeto = cms.bool(True)
        #   cut    = cms.double(5)
        #),
        hcal = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("patAODPhotonIsolations","gamIsoDepositHcalFromHits"), ## ..FromTowers if computed on AOD
            # parameters (E/gamma POG defaults)
            deltaR          = cms.double(0.4),
            skipDefaultVeto = cms.bool(True),
            # cut value - just a test, not an official one
            cut = cms.double(5.0)
        ),
        user = cms.VPSet(),
    ),

    markItems    = cms.bool(True), ## write the status flags in the output items
    bitsToIgnore = cms.vstring('Isolation/All'), ## Keep non-isolated photons
    saveRejected = cms.string(''), ## set this to a non empty label to save the list of items which fail
    saveAll      = cms.string(''), ## set this to a non empty label to save a list of all items both passing and failing
)


