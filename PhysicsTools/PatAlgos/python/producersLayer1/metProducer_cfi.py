import FWCore.ParameterSet.Config as cms

allLayer1METs = cms.EDProducer("PATMETProducer",
    # General configurables
    metSource  = cms.InputTag("allLayer0METs"),

    # Resolution configurables
    addResolutions   = cms.bool(True),
    metResoFile      = cms.string('PhysicsTools/PatUtils/data/Resolutions_met.root'),
    useNNResolutions = cms.bool(False), ## use the neural network approach?

    # Muon correction configurables
    addMuonCorrections = cms.bool(True),
    muonSource         = cms.InputTag("muons"), ## muon input source for corrections

    # Trigger matching configurables
    addTrigMatch  = cms.bool(True),
    trigPrimMatch = cms.VInputTag(cms.InputTag("metTrigMatchHLT1MET65")),

    # MC matching configurables
    addGenMET    = cms.bool(True),
    genMETSource = cms.InputTag("genMet"), ## GenMET source to be used for the matching

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

)


