import FWCore.ParameterSet.Config as cms

combinedSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
	jetTagComputer = cms.string('combinedSecondaryVertexV2Computer'),
	tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
	                         cms.InputTag("secondaryVertexTagInfos"))
)
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(combinedSecondaryVertexV2BJetTags,jetTagComputer = 'heavyIonCSVComputer')
