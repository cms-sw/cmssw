# configuration for Track Trigger emulation EDAnalyzer

import FWCore.ParameterSet.Config as cms

TrackerTFPAnalyzer_params = cms.PSet (

  UseMCTruth = cms.bool( True ), # enables analyze of TPs
  InputTagReconstructable = cms.InputTag("StubAssociator", "Reconstructable"), #
  InputTagSelection = cms.InputTag("StubAssociator", "UseForAlgEff"), #
  OutputLabelGP  = cms.string( "ProducerGP"  ),  #
  OutputLabelHT  = cms.string( "ProducerHT"  ),  #
  OutputLabelCTB = cms.string( "ProducerCTB" ),  #
  OutputLabelKF  = cms.string( "ProducerKF"  ),  #
  OutputLabelDR  = cms.string( "ProducerDR"  ),  #

)
