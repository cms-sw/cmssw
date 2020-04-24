import FWCore.ParameterSet.Config as cms

pfCombinedCvsLJetTags  = cms.EDProducer(
   "JetTagProducer",
   jetTagComputer = cms.string('charmTagsComputerCvsL'),
   tagInfos = cms.VInputTag(
      cms.InputTag('pfImpactParameterTagInfos'),
      cms.InputTag('pfInclusiveSecondaryVertexFinderCvsLTagInfos'),
      cms.InputTag('softPFMuonsTagInfos'),
      cms.InputTag('softPFElectronsTagInfos'),
      )
)

pfCombinedCvsBJetTags = pfCombinedCvsLJetTags.clone(
   jetTagComputer = cms.string('charmTagsComputerCvsB')
   )
