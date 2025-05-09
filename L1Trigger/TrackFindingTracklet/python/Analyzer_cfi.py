# configuration for hybrid track reconstruction chain analyzer

import FWCore.ParameterSet.Config as cms

TrackFindingTrackletAnalyzer_params = cms.PSet (

  UseMCTruth = cms.bool( True ), # enables analyze of TPs
  InputTagReconstructable = cms.InputTag("StubAssociator", "Reconstructable"), #
  InputTagSelection = cms.InputTag("StubAssociator", "UseForAlgEff"), #
  InputTag = cms.InputTag( "l1tTTTracksFromTrackletEmulation", "Level1TTTracks"), #
  OutputLabelTM  = cms.string  ( "ProducerTM"  ), #
  OutputLabelDR  = cms.string  ( "ProducerDR"  ), #
  OutputLabelKF  = cms.string  ( "ProducerKF"  ), #
  OutputLabelTQ  = cms.string  ( "ProducerTQ"  ), #
  OutputLabelTFP = cms.string  ( "ProducerTFP" ), #


)
