import FWCore.ParameterSet.Config as cms

hiRun2LegacyCSVv2JetTags  = cms.EDProducer(
   "JetTagProducer",
   jetTagComputer = cms.string('hiRun2LegacyCSVv2Tags'),
   tagInfos = cms.VInputTag(
      cms.InputTag('impactParameterTagInfos'),
      cms.InputTag('secondaryVertexFinderTagInfos'),
      )
)


