import FWCore.ParameterSet.Config as cms

demo = cms.EDAnalyzer('__class__'
     ,tracks = cms.untracked.InputTag('ctfWithMaterialTracks')
)
