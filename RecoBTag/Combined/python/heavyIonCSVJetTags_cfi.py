import FWCore.ParameterSet.Config as cms

heavyIonCSVJetTags  = cms.EDProducer(
   "JetTagProducer",
   jetTagComputer = cms.string('heavyIonCSVTags'),
   tagInfos = cms.VInputTag(
      cms.InputTag('impactParameterTagInfos'),
      cms.InputTag('secondaryVertexFinderTagInfos'),
      )
)
