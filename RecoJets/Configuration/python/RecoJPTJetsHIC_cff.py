# $Id: RecoJPTJetsHIC_cff.py,v 1.1 2010/03/11 15:17:33 srappocc Exp $
from RecoJets.JetPlusTracks.JetPlusTrackCorrectionsAA_cff import *

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIconePu5   = cms.Sequence(JetPlusTrackCorrectionsIconePu5)
jetPlusTrackZSPCorJetSisconePu5 = cms.Sequence(JetPlusTrackCorrectionsSisConePu5)
jetPlusTrackZSPCorJetAntiKtPu5  = cms.Sequence(JetPlusTrackCorrectionsAntiKtPu5)
 
#recoJPTJetsHIC=cms.Sequence(jetPlusTrackZSPCorJetIconePu5+jetPlusTrackZSPCorJetSisconePu5+jetPlusTrackZSPCorJetAntiKtPu5)
recoJPTJetsHIC=cms.Sequence(jetPlusTrackZSPCorJetIconePu5)
