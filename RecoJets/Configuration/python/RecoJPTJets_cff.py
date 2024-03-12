
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *
JetPlusTrackZSPCorJetAntiKt4.ptCUT = 15.

#define jetPlusTrackZSPCorJe  Task
jetPlusTrackZSPCorJetAntiKt4Task  = cms.Task(JetPlusTrackCorrectionsAntiKt4Task)

recoJPTJetsTask=cms.Task(jetPlusTrackZSPCorJetAntiKt4Task)
recoJPTJets=cms.Sequence(recoJPTJetsTask) 
# foo bar baz
# 5L5TZwrgjX5F1
