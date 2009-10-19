
# $Id: RecoJetsGR_cff.py,v 1.6 2009/08/13 15:07:46 srappocc Exp $
from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *
from JetMETCorrections.Configuration.ZSPJetCorrections219_cff import *

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIcone5   = cms.Sequence(ZSPJetCorrectionsIcone5+JetPlusTrackCorrectionsIcone5)
jetPlusTrackZSPCorJetSiscone5 = cms.Sequence(ZSPJetCorrectionsSisCone5+JetPlusTrackCorrectionsSisCone5)
jetPlusTrackZSPCorJetAntiKt5  = cms.Sequence(ZSPJetCorrectionsAntiKt5+JetPlusTrackCorrectionsAntiKt5)
 
recoJPTJets=cms.Sequence(jetPlusTrackZSPCorJetIcone5+jetPlusTrackZSPCorJetSiscone5+jetPlusTrackZSPCorJetAntiKt5)
