import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmLumiMonitor = DQMEDAnalyzer("DQMLumiMonitor",
    ModuleName          = cms.string('Info'),
    FolderName          = cms.string('Lumi'),
    PixelClusterInputTag = cms.InputTag('siPixelClusters'),
    LumiRecordName       = cms.string('expressLumiProducer'),                                 
    TH1ClusPar = cms.PSet(Xbins = cms.int32(150),Xmin = cms.double(0.5),Xmax = cms.double(7500.5)),
    TH1LumiPar = cms.PSet(Xbins = cms.int32(100),Xmin = cms.double(40000.0),Xmax = cms.double(90000.0)),                                
    TH1LSPar = cms.PSet(Xbins = cms.int32(2001),Xmin = cms.double(-0.5),Xmax = cms.double(2000.50))                                
)
