import FWCore.ParameterSet.Config as cms

# maintained in customisePhase2TTNoMC.py; the copy that was here still used the module labels from
# before the l1t renaming and failed
from L1Trigger.Configuration.customisePhase2TTNoMC import customisePhase2TTNoMC

def addHcalTriggerPrimitives(process):
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')

    return process

def addMenuNtuples(process):
    process.load("L1Trigger.L1TNtuples.l1PhaseIITreeStep1Producer_cfi")
    process.TFileService = cms.Service("TFileService",
        fileName = cms.string('L1NtuplePhaseII_Step1.root')
    )

    return process
