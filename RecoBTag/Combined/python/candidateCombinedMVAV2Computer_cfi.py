import FWCore.ParameterSet.Config as cms

candidateCombinedMVAV2Computer = cms.ESProducer("CombinedMVAV2JetTagESProducer",
	jetTagComputers = cms.vstring(
		'candidateJetProbabilityComputer',
		'candidateJetBProbabilityComputer',
		'candidateCombinedSecondaryVertexV2Computer',
		'softPFMuonComputer',
		'softPFElectronComputer'
	),
	weightFile = cms.FileInPath('RecoBTau/JetTagComputer/data/CombinedMVAV2.weights.xml')
)
