import FWCore.ParameterSet.Config as cms

updatedPatJets = cms.EDProducer("PATJetUpdater",
    # input
    jetSource = cms.InputTag("slimmedJets"),
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
    # jet energy corrections
    addJetCorrFactors    = cms.bool(True),
    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("updatedPatJetCorrFactors") ),
    # btag information
    addBTagInfo          = cms.bool(True),   ## master switch
    addDiscriminators    = cms.bool(True),   ## addition of btag discriminators
    discriminatorSources = cms.VInputTag(),
    # clone tag infos ATTENTION: these take lots of space!
    # usually the discriminators from the default algos
    # are sufficient
    addTagInfos     = cms.bool(False),
    tagInfoSources  = cms.VInputTag(),
    printWarning = cms.bool(True)
)


