# $Id: RecoJPTJets_cff.py,v 1.1 2009/10/19 17:53:21 stadie Exp $
from RecoJets.JetPlusTracks.JetPlusTrackCorrectionsAA_cff import *

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIconePu5   = cms.Sequence(JetPlusTrackCorrectionsIconePu5)
jetPlusTrackZSPCorJetSisconePu5 = cms.Sequence(JetPlusTrackCorrectionsSisConePu5)
jetPlusTrackZSPCorJetAntiKtPu5  = cms.Sequence(JetPlusTrackCorrectionsAntiKtPu5)
 
#recoJPTJetsHIC=cms.Sequence(jetPlusTrackZSPCorJetIconePu5+jetPlusTrackZSPCorJetSisconePu5+jetPlusTrackZSPCorJetAntiKtPu5)
recoJPTJetsHIC=cms.Sequence(jetPlusTrackZSPCorJetIconePu5)
