import FWCore.ParameterSet.Config as cms

def customisePhase2TTNoMC(process):
    process.L1TrackTrigger.replace(process.L1TPromptExtendedHybridTracksWithAssociators, process.L1TPromptExtendedHybridTracks)
    process.L1TrackTrigger.remove(process.TrackTriggerAssociatorClustersStubs)
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    return process
# foo bar baz
# teJdM8jf687ug
# 3z83qPDlci78j
