import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

pixelVTXMonitor = DQMEDAnalyzer("PixelVTXMonitor",
    ModuleName          = cms.string('BeamPixel'),
    FolderName          = cms.string('PixelVertex'),
    PixelClusterInputTag = cms.untracked.InputTag('siPixelClusters'),                                 
    PixelVertexInputTag = cms.untracked.InputTag('pixelVertices'),
    TH1ClusPar = cms.PSet(Xbins = cms.int32(150),Xmin = cms.double(0.5),Xmax = cms.double(7500.5)),
    TH1VtxPar  = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(-0.5),Xmax = cms.double(100.5)),
    HLTInputTag         = cms.untracked.InputTag('TriggerResults','','HLT'),
    MinVtxDoF           = cms.double(4.0),

#    HLTPathsOfInterest = cms.vstring('HLT_JetE30_NoBPTX_NoHalo_v8', 'HLT_JetE30_NoBPTX_v6', 'HLT_JetE50_NoBPTX3BX_NoHalo_v3', 'HLT_Physics_v2', 'HLT_PixelTracks_Multiplicity100_v6', 'HLT_PixelTracks_Multiplicity80_v6', 'HLT_ZeroBias_v4')
    HLTPathsOfInterest = cms.vstring('HLT_60Jet10', 'HLT_70Jet10','HLT_70Jet13', 'HLT_ZeroBias','HLT_Physics_v')                                 
)
