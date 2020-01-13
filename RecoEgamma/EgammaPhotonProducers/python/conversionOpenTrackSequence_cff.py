import FWCore.ParameterSet.Config as cms

import RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi

#producer from lowPt electronTracksOpen
gsfTracksOpenConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
   TrackProducer = cms.string('lowPtGsfEleGsfTracks'),
   setIsGsfTrackOpen = cms.bool(True),
   setArbitratedMergedEcalGeneral  = cms.bool(False),
   setArbitratedEcalSeeded  = cms.bool(False),
   setArbitratedMerged  = cms.bool(False),
   filterOnConvTrackHyp = cms.bool(False),
)


conversionOpenTrackTask = cms.Task(gsfTracksOpenConversionTrackProducer)


