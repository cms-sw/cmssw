import FWCore.ParameterSet.Config as cms

import RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi

#producer from lowPt electronTracksOpen
gsfTracksOpenConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
   TrackProducer                  = 'lowPtGsfEleGsfTracks',
   setIsGsfTrackOpen              = True,
   setArbitratedMergedEcalGeneral = False,
   setArbitratedEcalSeeded        = False,
   setArbitratedMerged            = False,
   filterOnConvTrackHyp           = False,
)

conversionOpenTrackTask = cms.Task(gsfTracksOpenConversionTrackProducer)
