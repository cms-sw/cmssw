
##____________________________________________________________________________||
caloType1CorrectedMet = cms.EDProducer("CorrectedCaloMETProducer",
    src = cms.InputTag('corMetGlobalMuons'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('caloJetMETcorr', 'type1')
    ),
    applyType2Corrections = cms.bool(False)
)

##____________________________________________________________________________||
caloType1p2CorrectedMet = cms.EDProducer("CorrectedCaloMETProducer",
    src = cms.InputTag('corMetGlobalMuons'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('caloJetMETcorr', 'type1')
    ),
    applyType2Corrections = cms.bool(True),
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('caloJetMETcorr', 'type2'),
        cms.InputTag('muonCaloMETcorr') # NOTE: use 'muonCaloMETcorr' for 'corMetGlobalMuons', do **not** use it for 'met' !!
    ),
    type2CorrFormula = cms.string("A + B*TMath::Exp(-C*x)"),
    type2CorrParameter = cms.PSet(
        A = cms.double(2.0),
        B = cms.double(1.3),
        C = cms.double(0.1)
    )
)

##____________________________________________________________________________||
produceCaloMETCorrections = cms.Sequence(
    caloJetMETcorr
   * muonCaloMETcorr
   * caloType1CorrectedMet
   * caloType1p2CorrectedMet
)

##____________________________________________________________________________||
