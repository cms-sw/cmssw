import FWCore.ParameterSet.Config as cms


pfIsolationCalculator = cms.PSet(
    #required inputs
    ComponentName = cms.string('pfIsolationCalculator'),
    particleType  = cms.int32(1), #photon
    coneDR        = cms.double(0.3),
    numberOfRings = cms.int32(1),
    ringSize      = cms.double(0.3),
    applyVeto     = cms.bool(True),
    applyPFPUVeto = cms.bool(True),
    applyDzDxyVeto = cms.bool(True),
    applyMissHitPhVeto = cms.bool(False),
    deltaRVetoBarrel = cms.bool(True),
    deltaRVetoEndcap = cms.bool(True),
    rectangleVetoBarrel = cms.bool(True),
    rectangleVetoEndcap = cms.bool(False),
    useCrystalSize      = cms.bool(True),
    checkClosestZVertex  = cms.bool(True),
#    
    deltaRVetoBarrelPhotons = cms.double(-1.0),
    deltaRVetoBarrelNeutrals = cms.double(-1.0),
    deltaRVetoBarrelCharged = cms.double(0.02),
    deltaRVetoEndcapPhotons = cms.double(0.07),
    deltaRVetoEndcapNeutrals = cms.double(-1.0),
    deltaRVetoEndcapCharged = cms.double(0.02),
#    
    rectangleDeltaPhiVetoBarrelPhotons = cms.double(1.),
    rectangleDeltaPhiVetoBarrelNeutrals = cms.double(-1),
    rectangleDeltaPhiVetoBarrelCharged = cms.double(-1),
    rectangleDeltaEtaVetoBarrelPhotons = cms.double(0.015),
    rectangleDeltaEtaVetoBarrelNeutrals = cms.double(-1),
    rectangleDeltaEtaVetoBarrelCharged = cms.double(-1),
#
    rectangleDeltaPhiVetoEndcapPhotons = cms.double(-1),
    rectangleDeltaPhiVetoEndcapNeutrals = cms.double(-1),
    rectangleDeltaPhiVetoEndcapCharged = cms.double(-1),
    rectangleDeltaEtaVetoEndcapPhotons = cms.double(-1),
    rectangleDeltaEtaVetoEndcapNeutrals = cms.double(-1),
    rectangleDeltaEtaVetoEndcapCharged = cms.double(-1),
    numberOfCrystalEndcapPhotons = cms.double(4.)

    
    
    
    
    
    


)


