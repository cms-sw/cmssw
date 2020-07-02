def customisePhase2TTNoMC(process):
    process.L1TrackTrigger.replace(process.L1PromptExtendedHybridTracksWithAssociators, process.L1PromptExtendedHybridTracks)
    process.L1TrackTrigger.remove(process.TrackTriggerAssociatorClustersStubs)
    return process
