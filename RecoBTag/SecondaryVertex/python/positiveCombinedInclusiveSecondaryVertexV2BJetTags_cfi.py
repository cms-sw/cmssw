import FWCore.ParameterSet.Config as cms

positiveCombinedInclusiveSecondaryVertexV2BJetTags = cms.EDProducer("JetTagProducer",
        jetTagComputer = cms.string('positiveCombinedSecondaryVertexV2Computer'),
        tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfos"),
                                 cms.InputTag("inclusiveSecondaryVertexFinderTagInfos"))
)
# foo bar baz
# F2PibYMDXFAa1
# dGs1vfNYvhVq8
