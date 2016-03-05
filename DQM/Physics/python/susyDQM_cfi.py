import FWCore.ParameterSet.Config as cms

#modules needed for jet flavor matching
jetPartons = cms.EDProducer("PartonSelector",
    withLeptons = cms.bool(False),
    src = cms.InputTag("genParticles")
)
ak4PFJetPartonAssociation = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("ak4PFJets"),
    partons = cms.InputTag("jetPartons"),
    coneSizeToAssociate = cms.double(0.3),
)
ak4PFJetFlavourAssociation = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("ak4PFJetPartonAssociation"),
    physicsDefinition = cms.bool(False)
)

susyDQM = cms.EDAnalyzer("RecoSusyDQM",

    muonCollection = cms.InputTag('muons'),
    electronCollection = cms.InputTag('gedGsfElectrons'),
    photonCollection = cms.InputTag('gedPhotons'),
    jetCollection = cms.InputTag('ak4PFJetsCHS'),
    metCollection = cms.InputTag('pfMet'),
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    conversions = cms.InputTag('conversions'),
    beamSpot = cms.InputTag('offlineBeamSpot'),
    fixedGridRhoFastjetAll = cms.InputTag('fixedGridRhoFastjetAll'),
    genParticles = cms.InputTag('genParticles'),
    genJets = cms.InputTag('ak4GenJets'),
    jetFlavourAssociation = cms.InputTag('ak4PFJetFlavourAssociation'),

    jetPtCut = cms.double(40),
    jetEtaCut = cms.double(3.0),
    jetTagCollection = cms.InputTag('pfCombinedInclusiveSecondaryVertexV2BJetTags'),
    csvv2Cut = cms.double(0.814), #medium CSVV2 working point

    #PHYS14 loose cuts-based electron ID
    elePtCut = cms.double(10),
    eleEtaCut = cms.double(2.5),
    eleMaxMissingHits = cms.int32(1),
    eleDEtaInCutBarrel = cms.double(0.012442),
    eleDPhiInCutBarrel = cms.double(0.072624),
    eleSigmaIetaIetaCutBarrel = cms.double(0.010557),
    eleHoverECutBarrel = cms.double(0.121476),
    eleD0CutBarrel = cms.double(0.022664),
    eleDZCutBarrel = cms.double(0.173670),
    eleOneOverEMinusOneOverPCutBarrel = cms.double(0.221803),
    eleRelIsoCutBarrel = cms.double(0.120026),
    eleDEtaInCutEndcap = cms.double(0.010654),
    eleDPhiInCutEndcap = cms.double(0.145129),
    eleSigmaIetaIetaCutEndcap = cms.double(0.032602),
    eleHoverECutEndcap = cms.double(0.131862),
    eleD0CutEndcap = cms.double(0.097358),
    eleDZCutEndcap = cms.double(0.198444),
    eleOneOverEMinusOneOverPCutEndcap = cms.double(0.142283),
    eleRelIsoCutEndcap = cms.double(0.162914),

    muPtCut = cms.double(10),
    muEtaCut = cms.double(2.4),
    muRelIsoCut = cms.double(0.2),

    #PHYS14 loose cuts-based photon ID
    phoPtCut = cms.double(20),
    phoEtaCut = cms.double(2.5),
    phoHoverECutBarrel = cms.double(0.048),
    phoSigmaIetaIetaCutBarrel = cms.double(0.0106),
    phoChHadIsoCutBarrel = cms.double(2.56),
    phoNeuHadIsoCutBarrel = cms.double(3.74),
    phoNeuHadIsoSlopeBarrel = cms.double(0.0025),
    phoPhotIsoCutBarrel = cms.double(2.68),
    phoPhotIsoSlopeBarrel = cms.double(0.001),
    phoHoverECutEndcap = cms.double(0.069),
    phoSigmaIetaIetaCutEndcap = cms.double(0.0266),
    phoChHadIsoCutEndcap = cms.double(3.12),
    phoNeuHadIsoCutEndcap = cms.double(17.11),
    phoNeuHadIsoSlopeEndcap = cms.double(0.0118),
    phoPhotIsoCutEndcap = cms.double(2.70),
    phoPhotIsoSlopeEndcap = cms.double(0.0059),

    useGen = cms.bool(True),
)

susyAnalyzer = cms.Path(jetPartons*ak4PFJetPartonAssociation*ak4PFJetFlavourAssociation*susyDQM)
