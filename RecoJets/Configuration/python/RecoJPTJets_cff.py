# $Id: RecoJPTJets_cff.py,v 1.4 2011/06/10 08:15:55 stadie Exp $
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIcone5   = cms.Sequence(JetPlusTrackCorrectionsIcone5)
jetPlusTrackZSPCorJetSiscone5 = cms.Sequence(JetPlusTrackCorrectionsSisCone5)
jetPlusTrackZSPCorJetAntiKt5  = cms.Sequence(JetPlusTrackCorrectionsAntiKt5)
 
recoJPTJets=cms.Sequence(jetPlusTrackZSPCorJetAntiKt5)
