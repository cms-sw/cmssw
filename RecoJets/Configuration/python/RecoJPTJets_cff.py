# $Id: RecoJPTJets_cff.py,v 1.5 2013/04/30 09:04:39 kodolova Exp $

from RecoJets.JetAssociationProducers.ak5JTA_cff import *
ak5JetTracksAssociatorAtVertexJPT = ak5JetTracksAssociatorAtVertex.clone()
ak5JetTracksAssociatorAtVertexJPT.useAssigned = cms.bool(True)
ak5JetTracksAssociatorAtVertexJPT.pvSrc = cms.InputTag("offlinePrimaryVertices")

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *
#JetPlusTrackCorrectionsAntiKt5.ptCUT = 15.

#define jetPlusTrackZSPCorJet sequences
jetPlusTrackZSPCorJetIcone5   = cms.Sequence(JetPlusTrackCorrectionsIcone5)
jetPlusTrackZSPCorJetSiscone5 = cms.Sequence(JetPlusTrackCorrectionsSisCone5)
jetPlusTrackZSPCorJetAntiKt5  = cms.Sequence(JetPlusTrackCorrectionsAntiKt5)
 
recoJPTJets=cms.Sequence(ak5JetTracksAssociatorAtVertexJPT*jetPlusTrackZSPCorJetAntiKt5)
