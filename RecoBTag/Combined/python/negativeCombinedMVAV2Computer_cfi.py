import FWCore.ParameterSet.Config as cms

negativeCombinedMVAV2Computer = cms.ESProducer("CombinedMVAV2JetTagESProducer",
	jetTagComputers = cms.vstring(
		'negativeOnlyJetProbabilityComputer',
		'negativeOnlyJetBProbabilityComputer',
		'negativeCombinedSecondaryVertexV2Computer',
		'negativeSoftPFMuonComputer',
		'negativeSoftPFElectronComputer'
	),
    mvaName = cms.string("bdt"),
	variables = cms.vstring(
        ["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"]
	),
	spectators = cms.vstring([]),
    useCondDB = cms.bool(False),
    useGBRForest = cms.bool(True),
    useAdaBoost = cms.bool(True),
	weightFile = cms.FileInPath('RecoBTag/Combined/data/CombinedMVAV2_13_07_2015.weights.xml.gz')
)
