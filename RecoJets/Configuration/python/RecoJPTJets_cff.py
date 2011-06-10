# $Id: RecoJPTJets_cff.py,v 1.3 2010/03/21 19:05:55 dlange Exp $
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIcone5   = cms.Sequence(JetPlusTrackCorrectionsIcone5)
jetPlusTrackZSPCorJetSiscone5 = cms.Sequence(JetPlusTrackCorrectionsSisCone5)
jetPlusTrackZSPCorJetAntiKt5  = cms.Sequence(JetPlusTrackCorrectionsAntiKt5)
 
recoJPTJets=cms.Sequence(jetPlusTrackZSPCorJetAntiKt5)
