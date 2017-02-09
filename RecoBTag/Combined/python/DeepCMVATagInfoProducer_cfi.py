import FWCore.ParameterSet.Config as cms

pfDeepCMVATagInfos = cms.EDProducer(
	'DeepCMVATagInfoProducer',
	deepNNTagInfos = cms.InputTag('pfDeepCSVTagInfos'),
	ipInfoSrc = cms.InputTag("pfImpactParameterTagInfos"),
	muInfoSrc = cms.InputTag("softPFMuonsTagInfos"),
	elInfoSrc = cms.InputTag("softPFElectronsTagInfos"),
	jpComputerSrc = cms.string('candidateJetProbabilityComputer'),
	jpbComputerSrc = cms.string('candidateJetBProbabilityComputer'),
	softmuComputerSrc = cms.string('softPFMuonComputer'),
	softelComputerSrc = cms.string('softPFElectronComputer'),
	cMVAPtThreshold = cms.double(200)
)
