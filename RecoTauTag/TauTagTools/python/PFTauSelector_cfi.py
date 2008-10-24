import FWCore.ParameterSet.Config as cms
pfTauSelector = cms.EDFilter("PFTauSelector",
   src = cms.InputTag("pfRecoTauProducer"),
   discriminators = cms.VPSet(
      cms.PSet( discriminator=cms.InputTag("pfRecoTauDiscriminationByIsolation"),selectionCut=cms.double(0.5))
   )
)


