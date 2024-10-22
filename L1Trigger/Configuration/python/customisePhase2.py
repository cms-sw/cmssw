import FWCore.ParameterSet.Config as cms

def customisePhase2TTNoMC(process):
    process.L1TrackTrigger.replace(process.L1PromptExtendedHybridTracksWithAssociators, process.L1PromptExtendedHybridTracks)
    process.L1TrackTrigger.remove(process.TrackTriggerAssociatorClustersStubs)
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    return process

def addHcalTriggerPrimitives(process):
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    return process

def addMenuNtuples(process):
    process.load("L1Trigger.L1TNtuples.l1PhaseIITreeStep1Producer_cfi")
    process.TFileService = cms.Service("TFileService",
        fileName = cms.string('L1NtuplePhaseII_Step1.root')
    )

    return process
