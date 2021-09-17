import FWCore.ParameterSet.Config as cms

def customisePhase2TTOn110(process):
    process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
    #when running directly, the ttclusterassoc uses the "mix" product name
    #however its ediased to simSiPixelDigis so its output with that name
    #so we have to adjust the input tag
    process.TTClusterAssociatorFromPixelDigis.digiSimLinks = cms.InputTag('simSiPixelDigis','Tracker')

    return process
