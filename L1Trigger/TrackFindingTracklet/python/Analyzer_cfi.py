# configuration for hybrid track reconstruction chain analyzer

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.Setup_cfi import TrackTrigger_params

TrackFindingTrackletAnalyzer_params = cms.PSet (

  UseMCTruth = cms.bool    (   True ), # enables analyze of TPs
  InputTag   = cms.InputTag( "", "" ), # to be costumized
  Process    = cms.string  (     "" ), # to be costumized
  NumChannel = cms.int32   (      1 ), # to be costumized
  NumRegions = TrackTrigger_params.DTC.NumRegions,
  NumLayers  = TrackTrigger_params.TrackFinding.NumLayers,

  LabelMC    = cms.string( "StubAssociator" ),

  OutputLabelTM  = cms.string( "ProducerTM"  ), #
  OutputLabelDR  = cms.string( "ProducerDR"  ), #
  OutputLabelKF  = cms.string( "ProducerKF"  ), #
  OutputLabelTQ  = cms.string( "ProducerTQ"  ), #
  OutputLabelTFP = cms.string( "ProducerTFP" ), #

)
