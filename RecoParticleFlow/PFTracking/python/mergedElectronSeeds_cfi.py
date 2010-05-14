import FWCore.ParameterSet.Config as cms



electronMergedSeeds =cms.EDProducer("ElectronSeedMerger",
     EcalBasedSeeds = cms.InputTag("ecalDrivenElectronSeeds"),
     TkBasedSeeds  = cms.InputTag("trackerDrivenElectronSeeds:SeedsForGsf")
    )

