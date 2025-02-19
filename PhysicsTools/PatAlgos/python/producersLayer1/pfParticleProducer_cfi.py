import FWCore.ParameterSet.Config as cms

patPFParticles = cms.EDProducer("PATPFParticleProducer",
    # General configurables
    pfCandidateSource = cms.InputTag("noJet"),

    # MC matching configurables
    addGenMatch = cms.bool(False),
    genParticleMatch = cms.InputTag(""),   ## particles source to be used for the MC matching
                                           ## must be an InputTag or VInputTag to a product of
                                           ## type edm::Association<reco::GenParticleCollection>
    embedGenMatch = cms.bool(False),       ## embed gen match inside the object instead of storing the ref

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

    # Efficiencies
    addEfficiencies = cms.bool(False),
    efficiencies    = cms.PSet(),

    # resolution
    addResolutions  = cms.bool(False),
    resolutions     = cms.PSet(),
)


