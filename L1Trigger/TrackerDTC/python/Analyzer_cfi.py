# configuration for AnalyzerDTC

import FWCore.ParameterSet.Config as cms

TrackerDTCAnalyzer_params = cms.PSet (

  UseMCTruth   = cms.bool( True ),                                 # eneables analyze of TPs
  InputTagMC   = cms.InputTag( "StubAssociator", "UseForEff"    ), # Assoc map
  InputTagReco = cms.InputTag( "ProducerDTC",    "StubAccepted" )  # DTC stubs

)
