import FWCore.ParameterSet.Config as cms

dqmLumiMonitor = cms.EDAnalyzer("DQMLumiMonitor",
    ModuleName          = cms.string('LuminosityInfo'),
    FolderName          = cms.string('EventInfo'),
    PixelClusterInputTag = cms.InputTag('siPixelClusters'),
    LumiRecordName       = cms.string('DIPLuminosityRcd'),                                 
    TH1ClusPar = cms.PSet(Xbins = cms.int32(150),Xmin = cms.double(0.5),Xmax = cms.double(7500.5)),
)
