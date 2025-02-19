import FWCore.ParameterSet.Config as cms

#Full Event content 
RecoPixelVertexingFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_pixelTracks_*_*', 
        'keep *_pixelVertices_*_*')
)
#RECO content
RecoPixelVertexingRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_pixelTracks_*_*', 
        'keep *_pixelVertices_*_*')
)

