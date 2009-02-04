# The following comments couldn't be translated into the new config version:

# Minimum Tk Pt
# inner cone veto (endcaps, |eta| >= 1.479)
import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import gamIsoFromDepsEcalFromHits,gamIsoFromDepsHcalFromTowers,gamIsoFromDepsTk

allLayer1Photons = cms.EDProducer("PATPhotonProducer",
    # General configurables
    photonSource = cms.InputTag("allLayer0Photons"),

                                  
    # user data to add
    userData = cms.PSet(
      # add custom classes here
      userClasses = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add doubles here
      userFloats = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add ints here
      userInts = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add "inline" functions here
      userFunctions = cms.vstring(""),
      userFunctionLabels = cms.vstring("")
    ),

    embedSuperCluster = cms.bool(True), ## whether to embed in AOD externally stored supercluster

    # Isolation configurables
    #   store isolation values
    isolation = cms.PSet(
        tracker = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("layer0PhotonIsolations","gamIsoDepositTk"),
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
            src = cms.InputTag("layer0PhotonIsolations","gamIsoDepositEcalFromHits"),
            # parameters (E/gamma POG defaults)
            vetos  = gamIsoFromDepsEcalFromHits.deposits[0].vetos,
            deltaR = gamIsoFromDepsEcalFromHits.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True),
#           # Or set your own vetos...
#            deltaR          = cms.double(0.4),
#            vetos           = cms.vstring('EcalBarrel:0.045', 'EcalEndcaps:0.070'),
        ),
        hcal = cms.PSet(
            # source IsoDeposit
            src = cms.InputTag("layer0PhotonIsolations","gamIsoDepositHcalFromTowers"),
            # parameters (E/gamma POG defaults)
            vetos  = gamIsoFromDepsHcalFromTowers.deposits[0].vetos,
            deltaR = gamIsoFromDepsHcalFromTowers.deposits[0].deltaR,
            skipDefaultVeto = cms.bool(True),
#           # Or set your own vetos...            
#            deltaR          = cms.double(0.4),
        ),
        user = cms.VPSet(),
    ),
    #   store isodeposits to recompute isolation
    isoDeposits = cms.PSet(
        tracker = cms.InputTag("layer0PhotonIsolations","gamIsoDepositTk"),
        ecal    = cms.InputTag("layer0PhotonIsolations","gamIsoDepositEcalFromHits"),
        hcal    = cms.InputTag("layer0PhotonIsolations","gamIsoDepositHcalFromTowers"),
    ),

    # PhotonID configurables
    addPhotonID = cms.bool(True),
    photonIDSource = cms.InputTag("layer0PhotonID"), ## ValueMap<reco::PhotonID> keyed to photonSource

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    trigPrimMatch = cms.VInputTag(cms.InputTag("photonTrigMatchHLT1PhotonRelaxed")),

    # MC matching configurables
    addGenMatch = cms.bool(True),
    embedGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag("photonMatch"), ## particles source to be used for the matching

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # Resolutions
    addResolutions  = cms.bool(False),
)


