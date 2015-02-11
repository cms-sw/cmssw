import FWCore.ParameterSet.Config as cms

negativeCombinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('negativeCombinedSecondaryVertexV2Computer'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexNegativeTagInfos"))
)

