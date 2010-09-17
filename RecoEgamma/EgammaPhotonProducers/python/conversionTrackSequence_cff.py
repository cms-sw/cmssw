import FWCore.ParameterSet.Config as cms

import RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi

import RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi

generalConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('generalTracks'),
    setTrackerOnly = cms.bool(True),
)

inOutConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('ckfInOutTracksFromConversions'),
    setArbitratedEcalSeeded = cms.bool(True),
)

outInConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('ckfOutInTracksFromConversions'),
    setArbitratedEcalSeeded = cms.bool(True),
)

gsfConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('electronGsfTracks'),
)

conversionTrackProducers = cms.Sequence(generalConversionTrackProducer*inOutConversionTrackProducer*outInConversionTrackProducer*gsfConversionTrackProducer)

inOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = cms.string('inOutConversionTrackProducer'),
    TrackProducer2 = cms.string('outInConversionTrackProducer'),
    #prefer collection settings:
    #-1: propagate output/flag from both input collections
    # 0: propagate output/flag from neither input collection
    # 1: arbitrate output/flag (remove duplicates by shared hits), give precedence to first input collection
    # 2: arbitrate output/flag (remove duplicates by shared hits), give precedence to second input collection
    # 3: arbitrate output/flag (remove duplicates by shared hits), arbitration first by number of hits, second by chisq/ndof  
    arbitratedEcalSeededPreferCollection = cms.int32(3),    
    arbitratedMergedPreferCollection = cms.int32(3),
)

generalInOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = cms.string('inOutOutInConversionTrackMerger'),
    TrackProducer2 = cms.string('generalConversionTrackProducer'),
    arbitratedMergedPreferCollection = cms.int32(1),
)

gsfGeneralInOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = cms.string('generalInOutOutInConversionTrackMerger'),
    TrackProducer2 = cms.string('gsfConversionTrackProducer'),
    arbitratedMergedPreferCollection = cms.int32(2),
)

conversionTrackMergers = cms.Sequence(inOutOutInConversionTrackMerger*generalInOutOutInConversionTrackMerger*gsfGeneralInOutOutInConversionTrackMerger)

conversionTrackSequence = cms.Sequence(conversionTrackProducers*conversionTrackMergers)
