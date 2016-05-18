import FWCore.ParameterSet.Config as cms

pfPositiveCombinedCvsLJetTags = cms.EDProducer(
   "JetTagProducer",
   jetTagComputer = cms.string('charmTagsPositiveComputerCvsL'),
   tagInfos = cms.VInputTag(
      cms.InputTag("pfImpactParameterTagInfos"),
      cms.InputTag("pfInclusiveSecondaryVertexFinderCvsLTagInfos"),
      cms.InputTag("softPFMuonsTagInfos"),
      cms.InputTag("softPFElectronsTagInfos")
      )
)
