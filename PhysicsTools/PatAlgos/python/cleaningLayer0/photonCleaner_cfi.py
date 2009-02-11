import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import gamIsoFromDepsEcalFromHits,gamIsoFromDepsHcalFromTowers,gamIsoFromDepsTk

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
            # cut value - just a test, not an official one
            cut = cms.double(3.0),
            # parameters (E/gamma POG defaults)
            vetos  = gamIsoFromDepsTk.deposits[0].vetos,
            deltaR = gamIsoFromDepsTk.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True), # This overrides previous settings
#           # Or set your own vetos...
#            deltaR = cms.double(0.3),              # Cone radius
#            vetos  = cms.vstring('0.015',          # Inner veto cone radius
#                                'Threshold(1.0)'), # Pt threshold
        ),
        ecal = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("patAODPhotonIsolations","gamIsoDepositEcalFromHits"), 
            # cut value - just a test, not an official one
            cut = cms.double(5.0),
            # parameters (E/gamma POG defaults)
            vetos  = gamIsoFromDepsEcalFromHits.deposits[0].vetos,
            deltaR = gamIsoFromDepsEcalFromHits.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True),
#           # Or set your own vetos...
#            deltaR          = cms.double(0.4),
#            vetos           = cms.vstring('EcalBarrel:0.045', 'EcalEndcaps:0.070'),
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
            src = cms.InputTag("patAODPhotonIsolations","gamIsoDepositHcalFromTowers"), ## 
            # cut value - just a test, not an official one
            cut = cms.double(5.0),
            # parameters (E/gamma POG defaults)
            vetos  = gamIsoFromDepsHcalFromTowers.deposits[0].vetos,
            deltaR = gamIsoFromDepsHcalFromTowers.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True),
#           # Or set your own vetos...            
#            deltaR          = cms.double(0.4),
        ),
        user = cms.VPSet(),
    ),

    markItems    = cms.bool(True), ## write the status flags in the output items
    bitsToIgnore = cms.vstring('Isolation/All'), ## Keep non-isolated photons
    saveRejected = cms.string(''), ## set this to a non empty label to save the list of items which fail
    saveAll      = cms.string(''), ## set this to a non empty label to save a list of all items both passing and failing
)


