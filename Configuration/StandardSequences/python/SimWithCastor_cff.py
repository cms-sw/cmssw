import FWCore.ParameterSet.Config as cms

def customise(process):
    process.g4SimHits.Generator.MinEtaCut =-7.0
    process.g4SimHits.Generator.MaxEtaCut = 5.5

    return(process)
