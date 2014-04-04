from DQM.PhysicsHWW.hwwAnalyzer_cfi import *
from DQM.PhysicsHWW.puJetIDAlgo_cff import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
from RecoBTag.Configuration.RecoBTag_cff import *

PFJetTracksAssociatorAtVertex = ic5PFJetTracksAssociatorAtVertex.clone()
PFJetTracksAssociatorAtVertex.jets = "ak5PFJets"
PFJetTracksAssociatorAtVertex.tracks = "generalTracks"
PFImpactParameterTagInfos = impactParameterTagInfos.clone()
PFImpactParameterTagInfos.jetTracks = "PFJetTracksAssociatorAtVertex"
PFTrackCountingHighEffBJetTags = trackCountingHighEffBJetTags.clone()
PFTrackCountingHighEffBJetTags.tagInfos = cms.VInputTag( cms.InputTag("PFImpactParameterTagInfos") )

SkipEvent = cms.untracked.vstring('ProductNotFound')

hwwDQM = cms.Sequence(PFJetTracksAssociatorAtVertex*PFImpactParameterTagInfos*
                      PFTrackCountingHighEffBJetTags*
                      hwwAnalyzer
                     )

hwwCosmicDQM = cms.Sequence(hwwAnalyzer)
