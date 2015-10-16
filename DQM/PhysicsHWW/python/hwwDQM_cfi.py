from DQM.PhysicsHWW.hwwAnalyzer_cfi import *
from RecoBTag.Configuration.RecoBTag_cff import *

PFImpactParameterTagInfos = pfImpactParameterTagInfos.clone()
PFImpactParameterTagInfos.jets = "ak4PFJets"
PFTrackCountingHighEffBJetTags = pfTrackCountingHighEffBJetTags.clone()
PFTrackCountingHighEffBJetTags.tagInfos = cms.VInputTag( cms.InputTag("PFImpactParameterTagInfos") )

SkipEvent = cms.untracked.vstring('ProductNotFound')

hwwDQM = cms.Sequence(PFImpactParameterTagInfos*
                      PFTrackCountingHighEffBJetTags*
                      hwwAnalyzer
                     )

hwwCosmicDQM = cms.Sequence(hwwAnalyzer)
