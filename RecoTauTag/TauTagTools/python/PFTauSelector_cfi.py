import FWCore.ParameterSet.Config as cms
pfTauSelector = cms.EDFilter("PFTauSelector",
   src = cms.InputTag("fixedConePFTauProducer"),
   discriminators = cms.VPSet(
      cms.PSet( discriminator=cms.InputTag("fixedConePFTauDiscriminationByIsolation"),selectionCut=cms.double(0.5))
   )
)


