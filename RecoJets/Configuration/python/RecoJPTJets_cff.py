# $Id: RecoJPTJets_cff.py,v 1.2 2010/03/09 03:30:46 srappocc Exp $
from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIcone5   = cms.Sequence(JetPlusTrackCorrectionsIcone5)
jetPlusTrackZSPCorJetSiscone5 = cms.Sequence(JetPlusTrackCorrectionsSisCone5)
jetPlusTrackZSPCorJetAntiKt5  = cms.Sequence(JetPlusTrackCorrectionsAntiKt5)
 
recoJPTJets=cms.Sequence(jetPlusTrackZSPCorJetIcone5*jetPlusTrackZSPCorJetAntiKt5)
