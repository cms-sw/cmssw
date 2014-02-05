
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *
JetPlusTrackZSPCorJetAntiKt4.ptCUT = 15.

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetAntiKt4  = cms.Sequence(JetPlusTrackCorrectionsAntiKt4)
 
recoJPTJets=cms.Sequence(jetPlusTrackZSPCorJetAntiKt4)
