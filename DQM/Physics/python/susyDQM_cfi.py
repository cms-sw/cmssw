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

    #Spring15 25ns loose cuts-based electron ID
    elePtCut = cms.double(10), # no update
    eleEtaCut = cms.double(2.5), # no update
    eleMaxMissingHitsBarrel = cms.int32(2), # 2 for Barrel, 1 for Endcap 
    eleDEtaInCutBarrel = cms.double(0.0105),
    eleDPhiInCutBarrel = cms.double(0.115),
    eleSigmaIetaIetaCutBarrel = cms.double(0.0103),
    eleHoverECutBarrel = cms.double(0.104),
    eleD0CutBarrel = cms.double(0.0261),
    eleDZCutBarrel = cms.double(0.41),
    eleOneOverEMinusOneOverPCutBarrel = cms.double(0.102),
    eleRelIsoCutBarrel = cms.double(0.0893),
    eleMaxMissingHitsEndcap = cms.int32(1), # 2 for Barrel, 1 for Endcap 
    eleDEtaInCutEndcap = cms.double(0.00814),
    eleDPhiInCutEndcap = cms.double(0.182),
    eleSigmaIetaIetaCutEndcap = cms.double(0.0301),
    eleHoverECutEndcap = cms.double(0.0897),
    eleD0CutEndcap = cms.double(0.118),
    eleDZCutEndcap = cms.double(0.822),
    eleOneOverEMinusOneOverPCutEndcap = cms.double(0.126),
    eleRelIsoCutEndcap = cms.double(0.121),

    muPtCut = cms.double(10), #no update
    muEtaCut = cms.double(2.4), #no update
    muRelIsoCut = cms.double(0.2), #no update

    #Spring15 25ns loose cuts-based photon ID
    phoPtCut = cms.double(20), #no update
    phoEtaCut = cms.double(2.5), #no update
    phoHoverECutBarrel = cms.double(0.05), 
    phoSigmaIetaIetaCutBarrel = cms.double(0.0102),
    phoChHadIsoCutBarrel = cms.double(3.32),
    phoNeuHadIsoCutBarrel = cms.double(1.92),#1.92 + 0.014*pho_pt + 0.000019*(pho_pt)2
    phoNeuHadIsoSlopeBarrel = cms.double(0.014),
    phoNeuHadIsoQuadraticBarrel = cms.double(0.000019),
    phoPhotIsoCutBarrel = cms.double(0.81),#0.81 + 0.0053*pho_pt
    phoPhotIsoSlopeBarrel = cms.double(0.0053),#
    phoHoverECutEndcap = cms.double(0.05),
    phoSigmaIetaIetaCutEndcap = cms.double(0.0274),
    phoChHadIsoCutEndcap = cms.double(1.97),
    phoNeuHadIsoCutEndcap = cms.double(11.86),#11.86 + 0.0139*pho_pt+0.000025*(pho_pt)2
    phoNeuHadIsoSlopeEndcap = cms.double(0.0139),
    phoNeuHadIsoQuadraticEndcap = cms.double(0.000025),
    phoPhotIsoCutEndcap = cms.double(0.83),#0.83 + 0.0034*pho_pt
    phoPhotIsoSlopeEndcap = cms.double(0.0034),

    useGen = cms.bool(True),
)

susyAnalyzer = cms.Path(jetPartons*ak4PFJetPartonAssociation*ak4PFJetFlavourAssociation*susyDQM)
