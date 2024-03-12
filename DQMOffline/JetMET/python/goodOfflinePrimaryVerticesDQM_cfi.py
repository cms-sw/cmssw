import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
goodOfflinePrimaryVerticesDQM = goodOfflinePrimaryVertices.clone() 
goodOfflinePrimaryVerticesDQMforMiniAOD = goodOfflinePrimaryVertices.clone(src = "offlineSlimmedPrimaryVertices")
# foo bar baz
# 6CE6dzy9vpuk6
