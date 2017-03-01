import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.PrimaryVertexMonitor_cff import pvMonitor

hltVerticesMonitoring = pvMonitor.clone()
hltVerticesMonitoring.beamSpotLabel = cms.InputTag("hltOnlineBeamSpot")

hltPixelVerticesMonitoring = hltVerticesMonitoring.clone()
hltPixelVerticesMonitoring.TopFolderName = cms.string("HLT/Vertexing/hltPixelVertices")
hltPixelVerticesMonitoring.vertexLabel   = cms.InputTag("hltPixelVertices")

Monitoring = hltVerticesMonitoring.clone()
Monitoring.TopFolderName = cms.string("HLT/Vertexing/")
Monitoring.vertexLabel   = cms.InputTag("")

hltVerticesPFFilterMonitoring = hltVerticesMonitoring.clone()
hltVerticesPFFilterMonitoring.TopFolderName = cms.string("HLT/Vertexing/hltVerticesPFFilter")
hltVerticesPFFilterMonitoring.vertexLabel   = cms.InputTag("hltVerticesPFFilter")

hltVerticesL3PFBjetsMonitoring = hltVerticesMonitoring.clone()
hltVerticesL3PFBjetsMonitoring.TopFolderName = cms.string("HLT/Vertexing/hltVerticesL3PFBjets")
hltVerticesL3PFBjetsMonitoring.vertexLabel   = cms.InputTag("hltVerticesL3PFBjets")

vertexingMonitorHLT = cms.Sequence(
    hltPixelVerticesMonitoring
    + hltTrimmedPixelVerticesMonitoring
    + hltVerticesPFFilterMonitoring
#    + hltVerticesL3PFBjets
)    

