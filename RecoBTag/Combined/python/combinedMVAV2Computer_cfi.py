import FWCore.ParameterSet.Config as cms

combinedMVAV2Computer = cms.ESProducer("CombinedMVAV2JetTagESProducer",
	jetTagComputers = cms.vstring(
		'jetProbabilityComputer',
		'jetBProbabilityComputer',
		'combinedSecondaryVertexV2Computer',
		'softPFMuonComputer',
		'softPFElectronComputer'
	),
	weightFile = cms.FileInPath('RecoBTag/Combined/data/CombinedMVAV2_13_07_2015.weights.xml.gz')
)
