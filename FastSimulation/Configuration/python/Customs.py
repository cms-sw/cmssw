#####################################
# a bunch of handy customisation functions
# author: Lukas Vanelderen
# date:   Jan 21 2015
#####################################

import FWCore.ParameterSet.Config as cms

def disableOOTPU(process):
    process.mix.maxBunch = cms.int32(0)
    process.mix.minBunch = cms.int32(0)
    # set the bunch spacing
    # bunch spacing matters for calorimeter calibration
    # by convention bunchspace is set to 450 in case of no oot pu
    process.mix.bunchspace = 450
    return process

# run this customisation function during the digi-step
# when processing a gen-sim sample that was generated with the HCALECAL geometry
def fakeSimHits_for_geometry_ECALHCAL(process):
    import FastSimulation.Validation.EmptySimHits_cfi
    process.g4SimHits = FastSimulation.Validation.EmptySimHits_cfi.emptySimHits.clone(
        pCaloHitInstanceLabels = ["CastorFI"],
        pSimHitInstanceLabels = []
        )
    for _entry  in process.mix.mixObjects.mixSH.input:
        process.g4SimHits.pSimHitInstanceLabels.append(_entry.getProductInstanceLabel())
    process.emptySimHits_step = cms.Path(process.g4SimHits)
    process.schedule.insert(0,process.emptySimHits_step)
    return process

def disableMaterialInteractionsTracker(process):
    process.famosSimHits.MaterialEffects.Bremsstrahlung = False
    process.famosSimHits.MaterialEffects.NuclearInteraction = False
    process.famosSimHits.MaterialEffects.PairProduction = False
    process.famosSimHits.MaterialEffects.MuonBremsstrahlung = False
    process.famosSimHits.MaterialEffects.MultipleScattering = False
    process.famosSimHits.MaterialEffects.EnergyLoss = False
    return process

