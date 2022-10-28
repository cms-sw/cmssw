import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.PATJetUpdater_cfi as _mod

updatedPatJets = _mod.PATJetUpdater.clone(
    # input
    jetSource = "slimmedJets",
    # add user data
    userData = dict(
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
      userFunctions = [],
      userFunctionLabels = []
    ),
    # sort
    sort                 = True,
    # jet energy corrections
    addJetCorrFactors    = True,
    jetCorrFactorsSource = ["updatedPatJetCorrFactors"],
    # btag information
    addBTagInfo          = True,   ## master switch
    addDiscriminators    = True,   ## addition of btag discriminators
    discriminatorSources = [],
    # clone tag infos ATTENTION: these take lots of space!
    # usually the discriminators from the default algos
    # are sufficient
    addTagInfos     = False,
    tagInfoSources  = [],
    printWarning    = True
)
