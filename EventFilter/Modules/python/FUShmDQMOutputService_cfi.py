import FWCore.ParameterSet.Config as cms


FUShmDQMOutputService = cms.Service( "FUShmDQMOutputService",
  initialMessageBufferSize = cms.untracked.int32( 1000000 ),
  lumiSectionsPerUpdate = cms.double( 1.0 ),
  useCompression = cms.bool( True ),
  compressionLevel = cms.int32( 1 ),
  lumiSectionInterval = cms.untracked.int32( 0 )
)

