import FWCore.ParameterSet.Config as cms

def customisePhase2TTNoMC(process):
    process.L1TrackTrigger.replace(process.L1TPromptExtendedHybridTracksWithAssociators, process.L1TPromptExtendedHybridTracks)
    process.L1TrackTrigger.remove(process.TrackTriggerAssociatorClustersStubs)
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    return process

def customiseL1P2NoMCTruth(process):
    """For the cmsDriver step L1P2 (standalone L1 configuration L1_Phase2_cff) run on stored events, e.g.
    GEN-SIM-DIGI-RAW: drops the MC-truth path (the tracker sim links read by the TT*Associator* modules
    are not stored) and loads the HCAL trigger primitive configuration, whose ES producers the L1
    emulation needs and the DIGI step otherwise provides."""
    process.schedule.remove(process.L1TrackTriggerMCTruth_step)
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    return process
