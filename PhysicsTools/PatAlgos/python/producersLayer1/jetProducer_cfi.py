import FWCore.ParameterSet.Config as cms

allLayer1Jets = cms.EDProducer("PATJetProducer",
    addJetCharge = cms.bool(True),
    addGenJetMatch = cms.bool(True),
    addAssociatedTracks = cms.bool(True),
    tagModuleLabelsToKeep = cms.vstring('combinedSVJetTags', 
        'combinedSVMVAJetTags', 
        'impactParameterMVAJetTags', 
        'jetProbabilityJetTags', 
        'softElectronJetTags', 
        'softMuonJetTags', 
        'softMuonNoIPJetTags', 
        'trackCountingHighEffJetTags', 
        'trackCountingHighPurJetTags'),
    partonJetSource = cms.InputTag("nonsenseName"),
    addGenPartonMatch = cms.bool(True),
    addBTagInfo = cms.bool(True),
    genPartonMatch = cms.InputTag("jetPartonMatch"),
    addPartonJetMatch = cms.bool(False),
    genJetMatch = cms.InputTag("jetGenJetMatch"),
    jetCharge = cms.PSet(
        var = cms.string('Pt'),
        exp = cms.double(1.0)
    ),
    caliBJetResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_bJets_MCJetCorJetIcone5.root'),
    JetPartonMapSource = cms.InputTag("jetPartonAssociation"),
    addJetTagRefs = cms.bool(True),
    jetSource = cms.InputTag("allLayer0Jets"),
    jetCorrFactorsSource = cms.InputTag("jetCorrFactors"),
    useNNResolutions = cms.bool(False),
    embedCaloTowers = cms.bool(True),
    tagModuleLabelPostfix = cms.string('Layer0'),
    addResolutions = cms.bool(True),
    getJetMCFlavour = cms.bool(True),
    addDiscriminators = cms.bool(True),
    trackAssociation = cms.PSet(
        deltaR = cms.double(0.5),
        maxNormChi2 = cms.double(5.0),
        minHits = cms.int32(8),
        tracksSource = cms.InputTag("generalTracks")
    ),
    caliJetResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_lJets_MCJetCorJetIcone5.root')
)


