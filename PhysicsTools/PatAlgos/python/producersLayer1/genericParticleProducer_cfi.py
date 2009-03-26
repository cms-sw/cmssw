import FWCore.ParameterSet.Config as cms

# NOTE: THIS IS JUST A SKELETON, YOU SHOULD FILL IT IN WITH "replace"

allLayer1GenericParticles = cms.EDProducer("PATGenericParticleProducer",
    ## Input (anything readable with View<Candidate>
    src = cms.InputTag("REPLACE_ME"),

    # AOD embedding
    embedTrack          = cms.bool(False),
    embedGsfTrack       = cms.bool(False), ## whether to embed in AOD externally stored gsf track
    embedStandAloneMuon = cms.bool(False), ## whether to embed in AOD externally stored standalone muon track
    embedCombinedMuon   = cms.bool(False), ## whether to embed in AOD externally stored combined muon track
    embedMultipleTracks = cms.bool(False), ## whether to embed in AOD externally stored multiple tracks (as per recoCandidate.track(int idx) )
    embedSuperCluster   = cms.bool(False), ## whether to embed in AOD externally stored supercluster
    embedCaloTower      = cms.bool(False), ## whether to embed in AOD externally stored calo tower

    # Isolation configurables
    isolation = cms.PSet(
    ),
    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(
    ),

    # any sort of "quality" value
    addQuality = cms.bool(False),
    qualitySource = cms.InputTag("REPLACE_ME"), ## must be ValueMap<float> associated to the input collection

    # Trigger matching configurables
    addTrigMatch = cms.bool(False),
    trigPrimMatch = cms.VInputTag(cms.InputTag("REPLACE_ME")), ## trigger primitive sources to be used for the matching

    # MC matching configurables
    addGenMatch = cms.bool(False),
    embedGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag("REPLACE_ME"), ## particles source to be used for the matching

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),
)


