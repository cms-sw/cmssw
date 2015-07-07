import FWCore.ParameterSet.Config as cms

combinedMVAV2Computer = cms.ESProducer("CombinedMVAV2JetTagESProducer",
	jetTagComputers = cms.vstring(
		'jetProbabilityComputer',
		'jetBProbabilityComputer',
		'combinedSecondaryVertexV2Computer',
		'softPFMuonComputer',
		'softPFElectronComputer'
	),
	weightFile = cms.FileInPath('RecoBTau/JetTagComputer/data/CombinedMVAV2.weights.xml')
)
