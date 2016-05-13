import FWCore.ParameterSet.Config as cms

pfNegativeCombinedCvsLJetTags = cms.EDProducer(
   "JetTagProducer",
   jetTagComputer = cms.string('charmTagsNegativeComputerCvsL'),
   tagInfos = cms.VInputTag(
      cms.InputTag("pfImpactParameterTagInfos"),
      cms.InputTag("pfInclusiveSecondaryVertexFinderCvsLNegativeTagInfos"),
      cms.InputTag("softPFMuonsTagInfos"),
      cms.InputTag("softPFElectronsTagInfos")
      )
)
