# The following comments couldn't be translated into the new config version:

# Minimum Tk Pt
# 
# Endcaps
# 
# set isolation but don't reject non-isolated electrons
import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff import eleIsoFromDepsEcalFromHits,eleIsoFromDepsHcalFromTowers,eleIsoFromDepsTk

allLayer0Electrons = cms.EDFilter("PATElectronCleaner",
    ## reco electron input source
    electronSource = cms.InputTag("pixelMatchGsfElectrons"), 

    # remove or flag duplicate (same track or same supercluster seed)
    removeDuplicates = cms.bool(True),

    # selection (e.g. ID)
    selection = cms.PSet(
        type = cms.string('none')
    ),

    # isolation (not mandatory, see bitsToIgnore below)
    isolation = cms.PSet(
        tracker = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("patAODElectronIsolations","eleIsoDepositTk"),
            # value for the cut (not optimized, just for testing)
            cut = cms.double(3.0),
            # parameters to compute isolation (Egamma POG defaults)
            vetos  = eleIsoFromDepsTk.deposits[0].vetos,
            deltaR = eleIsoFromDepsTk.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True), # This overrides previous settings
#           # Or set your own vetos...
#            deltaR = cms.double(0.3),
#            vetos = cms.vstring('0.015', # inner radius veto cone
#                'Threshold(1.0)'),       # threshold on individual track pt
        ),
        ecal = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("patAODElectronIsolations","eleIsoDepositEcalFromHits"), 
            # value for the cut (not optimized, just for testing)
            cut = cms.double(5.0),
            # parameters to compute isolation (Egamma POG defaults)
            vetos  = eleIsoFromDepsEcalFromHits.deposits[0].vetos,
            deltaR = eleIsoFromDepsEcalFromHits.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True), # This overrides previous settings
#           # Or set your own vetos...
#            deltaR = cms.double(0.4),
#            vetos = cms.vstring('EcalBarrel:0.040', 'EcalBarrel:RectangularEtaPhiVeto(-0.01,0.01,-0.5,0.5)',  # Barrel (|eta| < 1.479)
#                                'EcalEndcaps:0.070','EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'),
        ),
        hcal = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("patAODElectronIsolations","eleIsoDepositHcalFromTowers"), # FromTowers if computed from AOD
            # value for the cut (not optimized, just for testing)
            cut = cms.double(5.0),
            # parameters to compute isolation (Egamma POG defaults)
            vetos  = eleIsoFromDepsHcalFromTowers.deposits[0].vetos,
            deltaR = eleIsoFromDepsHcalFromTowers.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True),  # This overrides previous settings            
#           # Or set your own vetos...
#            deltaR = cms.double(0.4),
        ),
        user = cms.VPSet(),
    ),

    # duplicate removal configurables
    removeOverlaps = cms.PSet(
        ##  Somebody might want this feature (and possibly to add 'Overlap/Muons' to 'bitsToIgnore')
        # muons = cms.PSet(
        #    collection = cms.InputTag("allLayer0Muons")
        #    deltaR     = cms.double(0.3)
        # )
    ),

    markItems    = cms.bool(True), ## write the status flags in the output items
    bitsToIgnore = cms.vstring('Isolation/All'), ## keep non isolated electrons (but flag them)
                                   ## You can specify some bit names, e.g. "Overflow/User1", "Core/Duplicate", "Isolation/All".
    saveRejected = cms.string(''), ## set this to a non empty label to save the list of items which fail
    saveAll      = cms.string(''), ## set this to a non empty label to save a list of all items both passing and failing
)
