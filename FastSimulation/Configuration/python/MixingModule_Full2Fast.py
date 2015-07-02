#####################################
# customisation functions that allow to convert a FullSim PU cfg into a FastSim one
# main functions: prepareGenMixing and prepareDigiRecoMixing
# author: Lukas Vanelderen
# date:   Jan 21 2015
#####################################

import FWCore.ParameterSet.Config as cms


def digitizersFull2Fast(digitizers):

    # fastsim does not simulate castor
    if hasattr(digitizers,"castor"):
        delattr(digitizers,"castor")
    else:
        print "WARNING: digitizers has no attribute 'castor'"
        
    # fastsim does not digitize pixel and strip hits, it mixes tracks
    if hasattr(digitizers,"pixel") and hasattr(digitizers,"strip"):
        delattr(digitizers,"pixel")
        delattr(digitizers,"strip")
        import FastSimulation.Tracking.recoTrackAccumulator_cfi
        digitizers.tracker = cms.PSet(FastSimulation.Tracking.recoTrackAccumulator_cfi.recoTrackAccumulator)
    else:
        print "WARNING: digitizers has no attribute 'pixel' and/or 'strip'"
        print "       : => not mixing tracks"

    # fastsim has its own names for simhit collections
    for element in ["ecal","hcal"]:
        if hasattr(digitizers,element):
            getattr(digitizers,element).hitsProducer = "famosSimHits"
        else:
            print "WARNING: digitizers has no attribute '{0}'".format(element)
            
    # fastsim has different input for merged truth
    if hasattr(digitizers,"mergedtruth"):
        digitizers.mergedtruth.allowDifferentSimHitProcesses = True
        digitizers.mergedtruth.simHitCollections = cms.PSet(
            muon = cms.VInputTag( cms.InputTag('MuonSimHits','MuonDTHits'),
                                  cms.InputTag('MuonSimHits','MuonCSCHits'),
                                  cms.InputTag('MuonSimHits','MuonRPCHits') ),
            trackerAndPixel = cms.VInputTag( cms.InputTag('famosSimHits','TrackerHits') )
            )
        digitizers.mergedtruth.simTrackCollection = cms.InputTag('famosSimHits')
        digitizers.mergedtruth.simVertexCollection = cms.InputTag('famosSimHits')

    return digitizers


def prepareDigiRecoMixing(process):

    # switch to FastSim digitizers
    process.mix.digitizers = digitizersFull2Fast(process.mix.digitizers)

    # switch to FastSim mixObjects
    import FastSimulation.Configuration.mixObjects_cfi
    process.mix.mixObjects = FastSimulation.Configuration.mixObjects_cfi.theMixObjects

    # fastsim does not simulate castor
    # fastsim does not digitize pixel and strip hits
    for element in ["simCastorDigis","simSiPixelDigis","simSiStripDigis"]:
        if hasattr(process,element):
            delattr(process,element)
    
    # get rid of some FullSim specific psets that work confusing when dumping FastSim cfgs 
    # (this is optional)
    del process.theDigitizers
    del process.theDigitizersValid    
    del process.trackingParticles
    del process.stripDigitizer
    del process.SiStripSimBlock
    del process.castorDigitizer
    del process.pixelDigitizer
    del process.ecalDigitizer
    
    # get rid of FullSim specific services that work confusing when dumping FastSim cfgs
    # (this is optional)
    del process.siStripGainSimESProducer

    return process
