import FWCore.ParameterSet.Config as cms

# NOTE: THIS IS JUST A SKELETON, YOU SHOULD FILL IT IN WITH "replace"

patGenericParticles = cms.EDProducer("PATGenericParticleProducer",
    ## Input (anything readable with View<Candidate>
    src = cms.InputTag("REPLACE_ME"),

    # add user data
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
      # add candidate ptrs here
      userCands = cms.PSet(
        src = cms.VInputTag('')
      ),
      # add "inline" functions here
      userFunctions = cms.vstring(),
      userFunctionLabels = cms.vstring()
    ),

    # AOD embedding
    embedTrack          = cms.bool(False),
    embedGsfTrack       = cms.bool(False), ## whether to embed in AOD externally stored gsf track
    embedStandAloneMuon = cms.bool(False), ## whether to embed in AOD externally stored standalone muon track
    embedCombinedMuon   = cms.bool(False), ## whether to embed in AOD externally stored combined muon track
    embedMultipleTracks = cms.bool(False), ## whether to embed in AOD externally stored multiple tracks (as per recoCandidate.track(int idx) )
    embedSuperCluster   = cms.bool(False), ## whether to embed in AOD externally stored supercluster
    embedCaloTower      = cms.bool(False), ## whether to embed in AOD externally stored calo tower

    # embed IsoDeposits to recompute isolation
    isoDeposits = cms.PSet(
    ),

    # user defined isolation variables the variables defined here will be accessible
    # via pat::GenericParticle::userIsolation(IsolationKeys key) with the key as defined in
    # DataFormats/PatCandidates/interface/Isolation.h
    userIsolation = cms.PSet(
    ),
                                           
    # any sort of "quality" value
    addQuality = cms.bool(False),
    qualitySource = cms.InputTag("REPLACE_ME"), ## must be ValueMap<float> associated to the input collection

    # MC matching configurables
    addGenMatch = cms.bool(False),
    embedGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag("REPLACE_ME"), ## particles source to be used for the matching

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution
    addResolutions  = cms.bool(False),
    resolutions     = cms.PSet(),
)
