import FWCore.ParameterSet.Config as cms

candidateNegativeCombinedMVAV2Computer = cms.ESProducer("CombinedMVAV2JetTagESProducer",
    gbrForestLabel = cms.string('btag_CombinedMVAv2_BDT'),
    jetTagComputers = cms.vstring(
        'candidateNegativeOnlyJetProbabilityComputer',
        'candidateNegativeOnlyJetBProbabilityComputer',
        'candidateNegativeCombinedSecondaryVertexV2Computer',
        'negativeSoftPFMuonComputer',
        'negativeSoftPFElectronComputer'
    ),
    mvaName = cms.string('bdt'),
    spectators = cms.vstring(),
    useAdaBoost = cms.bool(False),
    useCondDB = cms.bool(True),
    useGBRForest = cms.bool(True),
    variables = cms.vstring(
        'Jet_CSV',
        'Jet_CSVIVF',
        'Jet_JP',
        'Jet_JBP',
        'Jet_SoftMu',
        'Jet_SoftEl'
    ),
    weightFile = cms.FileInPath('RecoBTag/Combined/data/CombinedMVAV2_13_07_2015.weights.xml.gz')
)
