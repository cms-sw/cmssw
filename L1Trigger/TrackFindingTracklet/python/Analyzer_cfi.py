# ParameterSet used by AnalyzerKFin and AnalyzerTracklet
import FWCore.ParameterSet.Config as cms

TrackFindingTrackletAnalyzer_params = cms.PSet (

  UseMCTruth = cms.bool( True ), # enables analyze of TPs
  InputTagReconstructable = cms.InputTag("StubAssociator", "Reconstructable"), #
  InputTagSelection = cms.InputTag("StubAssociator", "UseForAlgEff"), #


)