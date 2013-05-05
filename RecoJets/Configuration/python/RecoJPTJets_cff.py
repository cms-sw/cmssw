# $Id: RecoJPTJets_cff.py,v 1.6 2013/05/01 20:46:01 srappocc Exp $

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *
JetPlusTrackCorrectionsAntiKt5.ptCUT = 15.

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIcone5   = cms.Sequence(JetPlusTrackCorrectionsIcone5)
jetPlusTrackZSPCorJetSiscone5 = cms.Sequence(JetPlusTrackCorrectionsSisCone5)
jetPlusTrackZSPCorJetAntiKt5  = cms.Sequence(JetPlusTrackCorrectionsAntiKt5)
 
recoJPTJets=cms.Sequence(ak5JetTracksAssociatorAtVertexJPT*jetPlusTrackZSPCorJetAntiKt5)
