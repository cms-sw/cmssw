import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.METSignificanceParams_cfi import METSignificanceParams


patMETs = cms.EDProducer("PATMETProducer",
    # input
    metSource  = cms.InputTag("pfMetT1"),

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
    addMuonCorrections = cms.bool(False),
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

    # significance
    computeMETSignificance  = cms.bool(False),
    # significance computation parameters, not used
    # if the significance is not computed
    srcJets = cms.InputTag("cleanedPatJets"),
    srcPFCands =  cms.InputTag("particleFlow"),
    srcLeptons = cms.VInputTag("selectedPatElectrons", "selectedPatMuons", "selectedPatPhotons"),
    srcJetSF = cms.string('AK4PFchs'),
    srcJetResPt = cms.string('AK4PFchs_pt'),
    srcJetResPhi = cms.string('AK4PFchs_phi'),
    srcRho = cms.InputTag('fixedGridRhoAll'),
    parameters = METSignificanceParams
)
