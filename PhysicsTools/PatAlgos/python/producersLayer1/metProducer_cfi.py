import FWCore.ParameterSet.Config as cms

layer1METs = cms.EDProducer("PATMETProducer",
    # General configurables
    metSource  = cms.InputTag("corMetType1Icone5Muons"),

                               
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

    # Resolution configurables
    addResolutions   = cms.bool(False),

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


