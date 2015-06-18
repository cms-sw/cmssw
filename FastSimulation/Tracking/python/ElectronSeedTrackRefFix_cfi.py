import FWCore.ParameterSet.Config as cms

fixedTrackerDrivenElectronSeeds = cms.EDProducer(
    "ElectronSeedTrackRefFix",
    PreGsfLabel = cms.string("SeedsForGsf"),
    PreIdLabel = cms.string("preid"),
    oldTrackCollection = cms.InputTag("generalTracksBeforeMixing"),
    newTrackCollection = cms.InputTag("generalTracks"),
    seedCollection = cms.InputTag("trackerDrivenElectronSeeds","SeedsForGsf"),
    idCollection = cms.InputTag("trackerDrivenElectronSeeds","preid")
    )
