import FWCore.ParameterSet.Config as cms

pixelVTXMonitor = cms.EDAnalyzer("PixelVTXMonitor",
    ModuleName          = cms.string('BeamPixel'),
    FolderName          = cms.string('PixelVertex'),
    PixelVertexInputTag = cms.InputTag('pixelVertices'),
    HLTInputTag         = cms.InputTag('TriggerResults','','HLT'),
    MinVtxDoF           = cms.double(4.0),
    HLTPathsOfInterest = cms.vstring('HLT_ZeroBias_v3','HLT_JetE30_NoBPTX_v3','HLT_Physics_v1')
                                             
)
