import FWCore.ParameterSet.Config as cms

# tracking monitor
from DQMOffline.Trigger.PrimaryVertexMonitoring_cff import *

vertexingMonitorHLTsequence = cms.Sequence(
    hltPixelVerticesMonitoring
    + hltVerticesPFFilterMonitoring
#    + hltVerticesL3PFBjets
)    
