import FWCore.ParameterSet.Config as cms

TrackerTFPAnalyzer_params = cms.PSet (

  UseMCTruth = cms.bool( True ), # enables analyze of TPs
  InputTagReconstructable = cms.InputTag("StubAssociator", "Reconstructable"), #
  InputTagSelection = cms.InputTag("StubAssociator", "UseForAlgEff"), #


)