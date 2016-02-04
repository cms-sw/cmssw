import FWCore.ParameterSet.Config as cms

candidateNegativeCombinedMVAV2Computer = cms.ESProducer("CombinedMVAV2JetTagESProducer",
	jetTagComputers = cms.vstring(
		'candidateNegativeOnlyJetProbabilityComputer',
		'candidateNegativeOnlyJetBProbabilityComputer',
		'candidateNegativeCombinedSecondaryVertexV2Computer',
		'negativeSoftPFMuonComputer',
		'negativeSoftPFElectronComputer'
	),
	mvaName = cms.string("bdt"),
	variables = cms.vstring(
		["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"]
	),
	spectators = cms.vstring([]),
	useCondDB = cms.bool(True),
	gbrForestLabel = cms.string("btag_CombinedMVAv2_BDT"),
	useGBRForest = cms.bool(True),
	useAdaBoost = cms.bool(False),
	weightFile = cms.FileInPath('RecoBTag/Combined/data/CombinedMVAV2_13_07_2015.weights.xml.gz')
)
