import FWCore.ParameterSet.Config as cms

allLayer1Taus = cms.EDProducer("PATTauProducer",
    # General configurables
    tauSource = cms.InputTag("allLayer0Taus"),

    embedLeadTrack       = cms.bool(False), ## whether to embed in AOD externally stored leading track
    embedSignalTracks    = cms.bool(False), ## whether to embed in AOD externally stored signal tracks
    embedIsolationTracks = cms.bool(False), ## whether to embed in AOD externally stored isolation tracks

    # resolution configurables
    addResolutions = cms.bool(True),
    tauResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_tau.root'),
    useNNResolutions = cms.bool(True), ## use the neural network approach?

    # isolation configurables
    isolation = cms.PSet(),
    isoDeposits = cms.PSet(),

    # Trigger matching configurables
    addTrigMatch = cms.bool(True),
    trigPrimMatch = cms.VInputTag(cms.InputTag("tauTrigMatchHLT1Tau")),

    # MC matching configurables
    addGenMatch = cms.bool(True),
    genParticleMatch = cms.InputTag("tauMatch"), ## particles source to be used for the matching

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

)


