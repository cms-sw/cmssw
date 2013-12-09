import FWCore.ParameterSet.Config as cms

positiveCombinedMVA = cms.ESProducer("CombinedMVAJetTagESProducer",
	useCategories = cms.bool(False),
	calibrationRecord = cms.string('CombinedMVA'),
	jetTagComputers = cms.VPSet(
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveOnlyJetProbability')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('combinedSecondaryVertexPositive')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveSoftPFMuon')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveSoftPFElectron')
		)
	)
)

positiveCombinedSecondaryVertexSoftPFLeptonV1 = cms.ESProducer("CombinedMVAJetTagESProducer",
	useCategories = cms.bool(False),
	calibrationRecord = cms.string('CombinedCSVSL'),
	jetTagComputers = cms.VPSet(
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveOnlyJetProbability')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('combinedSecondaryVertexV1Positive')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveSoftPFMuon')
		),
		cms.PSet(
			discriminator = cms.bool(True),
			variables = cms.bool(False),
			jetTagComputer = cms.string('positiveSoftPFElectron')
		)
	)
)
