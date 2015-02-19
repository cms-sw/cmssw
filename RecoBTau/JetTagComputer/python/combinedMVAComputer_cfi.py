import FWCore.ParameterSet.Config as cms

combinedMVAComputer = cms.ESProducer("CombinedMVAJetTagESProducer",
	useCategories = cms.bool(False),
	calibrationRecord = cms.string('CombinedMVA'),
	jetTagComputers = cms.VPSet(
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('jetProbabilityComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('combinedSecondaryVertexComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFMuonComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('softPFElectronComputer')
		)
	)
)

