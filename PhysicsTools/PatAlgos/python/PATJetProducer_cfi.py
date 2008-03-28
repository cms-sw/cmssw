# The following comments couldn't be translated into the new config version:

# the JetTag labels that should be considered and kept (excluding tagModuleLabelPostfix)
import FWCore.ParameterSet.Config as cms

allLayer1Jets = cms.EDProducer("PATJetProducer",
    # Jet charge configurables
    addJetCharge = cms.bool(True),
    addGenJetMatch = cms.bool(True), ## switch on/off matching to GenJet's

    # track association configurables
    addAssociatedTracks = cms.bool(True),
    tagModuleLabelsToKeep = cms.vstring('combinedSVJetTags', 'combinedSVMVAJetTags', 'impactParameterMVAJetTags', 'jetProbabilityJetTags', 'softElectronJetTags', 'softMuonJetTags', 'softMuonNoIPJetTags', 'trackCountingHighEffJetTags', 'trackCountingHighPurJetTags'),
    partonJetSource = cms.InputTag("nonsenseName"), ## ParticleJet source to be used for the matching

    # MC matching configurables
    addGenPartonMatch = cms.bool(True),
    # input root file for the resolution functions
    # b-Tag info configurables
    addBTagInfo = cms.bool(True),
    genPartonMatch = cms.InputTag("jetPartonMatch"), ## particles source to be used for the matching

    addPartonJetMatch = cms.bool(False), ## switch on/off matching to PartonJet's (not implemented yet)

    genJetMatch = cms.InputTag("jetGenJetMatch"), ## GenJet source to be used for the matching

    jetCharge = cms.PSet( ## the jet charge parameter set

        var = cms.string('Pt'),
        exp = cms.double(1.0)
    ),
    # input root file for the resolution functions
    caliBJetResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_bJets_MCJetCorJetIcone5.root'),
    JetPartonMapSource = cms.InputTag("jetPartonAssociation"), ## the match-collection, produced by default from PATHighLevelReco.cff

    addJetTagRefs = cms.bool(True), ## switch on/off the addition of references to the b-tag JetTag's

    # General configurables
    jetSource = cms.InputTag("allLayer0Jets"),
    # Jet energy scale correction configurables
    jetCorrFactorsSource = cms.InputTag("jetCorrFactors"),
    useNNResolutions = cms.bool(False), ## use the neural network approach?

    tagModuleLabelPostfix = cms.string('Layer0'), ## strip this string from the end of the tag module labels

    # Resolution configurables
    addResolutions = cms.bool(True),
    # Jet flavour idetification configurables
    getJetMCFlavour = cms.bool(True),
    addDiscriminators = cms.bool(True), ## switch on/off the addition of the btag discriminators

    trackAssociation = cms.PSet( ## the track association parameter set

        deltaR = cms.double(0.5),
        maxNormChi2 = cms.double(5.0),
        minHits = cms.int32(8),
        tracksSource = cms.InputTag("ctfWithMaterialTracks")
    ),
    caliJetResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_lJets_MCJetCorJetIcone5.root')
)


