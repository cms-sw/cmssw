import FWCore.ParameterSet.Config as cms

def customise(process):
    process.load("SimG4Core.Application.g4SimHits_cfi")
    process.g4SimHits.Generator.MinEtaCut =-7.0
    process.g4SimHits.Generator.MaxEtaCut = 5.5
    process.load('SimCalorimetry.CastorSim.castordigi_cfi') 
    process.simCastorDigis.castor.photoelectronsToAnalog = 4.24

    return(process)
