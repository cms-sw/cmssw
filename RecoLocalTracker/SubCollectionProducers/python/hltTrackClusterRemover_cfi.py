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
      minGoodStripCharge = cms.double( 0.0) ## this value is coherent w/ any cuts apply (standard value is ~60)
                                            ## for more details, look @https://indico.cern.ch/getFile.py/access?contribId=1&resId=0&materialId=slides&confId=292571 (for instance)
   ),
   doStripChargeCheck = cms.bool(False),
)
