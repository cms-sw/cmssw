import FWCore.ParameterSet.Config as cms

conversionTrackMerger = cms.EDProducer("ConversionTrackMerger",
    # minimum shared fraction to be called duplicate
    ShareFrac = cms.double(0.19),
    TrackProducer1 = cms.string(''),
    TrackProducer2 = cms.string(''),
    allowFirstHitShare = cms.bool(True),
    checkCharge = cms.bool(True),
    minProb = cms.double(1e-6),
    #prefer collection settings:
    #-1: propagate output/flag from both input collections
    # 0: propagate output/flag from neither input collection
    # 1: arbitrate output/flag (remove duplicates by shared hits), give precedence to first input collection
    # 2: arbitrate output/flag (remove duplicates by shared hits), give precedence to second input collection
    # 3: arbitrate output/flag (remove duplicates by shared hits), arbitration first by number of hits, second by chisq/ndof   
    outputPreferCollection = cms.int32(-1),
    trackerOnlyPreferCollection = cms.int32(-1),
    arbitratedEcalSeededPreferCollection = cms.int32(-1),
    arbitratedMergedPreferCollection = cms.int32(3),
    arbitratedMergedEcalGeneralPreferCollection = cms.int32(-1),    
)
