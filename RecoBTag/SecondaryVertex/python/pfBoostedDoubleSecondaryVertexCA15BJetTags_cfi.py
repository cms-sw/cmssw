import FWCore.ParameterSet.Config as cms

pfBoostedDoubleSecondaryVertexCA15BJetTags = cms.EDProducer("JetTagProducer",
    jetTagComputer = cms.string('candidateBoostedDoubleSecondaryVertexCA15Computer'),
    tagInfos = cms.VInputTag(cms.InputTag("pfBoostedDoubleSVCA15TagInfos"))
)
# foo bar baz
# iuzcZv9hiQho4
# oj8gJixRPFKNQ
