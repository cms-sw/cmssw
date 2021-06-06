import FWCore.ParameterSet.Config as cms

def customisePhase2TTNoMC(process):
    process.L1TrackTrigger.replace(process.L1PromptExtendedHybridTracksWithAssociators, process.L1PromptExtendedHybridTracks)
    process.L1TrackTrigger.remove(process.TrackTriggerAssociatorClustersStubs)
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    return process
