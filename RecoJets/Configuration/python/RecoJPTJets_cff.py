# $Id: RecoJPTJets_cff.py,v 1.9 2013/05/21 11:16:38 speer Exp $

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *
JetPlusTrackZSPCorJetAntiKt5.ptCUT = 15.

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIcone5   = cms.Sequence(JetPlusTrackCorrectionsIcone5)
jetPlusTrackZSPCorJetSiscone5 = cms.Sequence(JetPlusTrackCorrectionsSisCone5)
jetPlusTrackZSPCorJetAntiKt5  = cms.Sequence(JetPlusTrackCorrectionsAntiKt5)
 
recoJPTJets=cms.Sequence(jetPlusTrackZSPCorJetAntiKt5)
