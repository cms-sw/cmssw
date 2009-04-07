import FWCore.ParameterSet.Config as cms

layer1METs = cms.EDProducer("PATMETProducer",
    # input 
    metSource  = cms.InputTag("corMetType1Icone5Muons"),

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
      # add "inline" functions here
      userFunctions = cms.vstring(),
      userFunctionLabels = cms.vstring()
    ),

    # muon correction
    addMuonCorrections = cms.bool(True),
    muonSource         = cms.InputTag("muons"),

    # trigger matching
    addTrigMatch  = cms.bool(False),
    trigPrimMatch = cms.VInputTag(''),

    # mc matching configurables
    addGenMET    = cms.bool(True),
    genMETSource = cms.InputTag("genMetCalo"),

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution
    addResolutions  = cms.bool(False),
)


