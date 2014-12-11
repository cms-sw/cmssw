import FWCore.ParameterSet.Config as cms

positiveCombinedMVAComputer = cms.ESProducer("CombinedMVAJetTagESProducer",
	useCategories = cms.bool(False),
	calibrationRecord = cms.string('CombinedMVA'),
	jetTagComputers = cms.VPSet(
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveOnlyJetProbabilityComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveCombinedSecondaryVertexComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveSoftPFMuonComputer')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveSoftPFElectronComputer')
		)
	)
)

