import FWCore.ParameterSet.Config as cms

patMETs = cms.EDProducer("PATMETProducer",
    # input 
    metSource  = cms.InputTag("caloType1CorrectedMet"),

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

    # muon correction
    addMuonCorrections = cms.bool(True),
    muonSource         = cms.InputTag("muons"),

    # mc matching configurables
    addGenMET    = cms.bool(True),
    genMETSource = cms.InputTag("genMetTrue"),

    # efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution
    addResolutions  = cms.bool(False),
    resolutions     = cms.PSet(),
)


