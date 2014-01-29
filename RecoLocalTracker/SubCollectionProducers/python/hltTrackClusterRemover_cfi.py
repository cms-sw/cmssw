import FWCore.ParameterSet.Config as cms

hltTrackClusterRemover = cms.EDProducer( "HLTTrackClusterRemover",
   trajectories          = cms.InputTag( "hltPFlowTrackSelectionHighPurity" ),
   doStrip               = cms.bool( True ),
   doPixel               = cms.bool( True ),
   stripClusters         = cms.InputTag( "hltSiStripRawToClustersFacility" ),
   pixelClusters         = cms.InputTag( "hltSiPixelClusters" ),
   oldClusterRemovalInfo = cms.InputTag( "" ),
   Common = cms.PSet(
      maxChi2            = cms.double( 9.0 ),
      minGoodStripCharge = cms.double(60.0)
   ),
   doStripChargeCheck = cms.bool(True),
)
