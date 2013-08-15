import FWCore.ParameterSet.Config as cms
import copy

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *
TCTauJetPlusTrackZSPCorJetAntiKt5 = copy.deepcopy(JetPlusTrackZSPCorJetAntiKt5)
TCTauJetPlusTrackZSPCorJetAntiKt5.UseZSP = cms.bool(False)
TCTauJetPlusTrackZSPCorJetAntiKt5.ResponseMap   = 'CondFormats/JetMETObjects/data/CMSSW_362_resptowers.txt'
TCTauJetPlusTrackZSPCorJetAntiKt5.UseEfficiency = cms.bool(False)

from RecoTauTag.RecoTau.CaloRecoTauTagInfoProducer_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauProducer_cfi import *
caloRecoTauTagInfoProducer.CaloJetTracksAssociatorProducer = cms.InputTag('JPTAntiKt5JetTracksAssociatorAtVertex')
JPTCaloRecoTauProducer = copy.deepcopy(caloRecoTauProducer)

# Anti-Kt
from RecoJets.JetAssociationProducers.ak4JTA_cff import*

JPTAntiKt5JetTracksAssociatorAtVertex = ak4JetTracksAssociatorAtVertex.clone()
JPTAntiKt5JetTracksAssociatorAtVertex.jets = cms.InputTag("TCTauJetPlusTrackZSPCorJetAntiKt5")

JPTAntiKt5JetTracksAssociatorAtCaloFace = ak4JetTracksAssociatorAtCaloFace.clone()
JPTAntiKt5JetTracksAssociatorAtCaloFace.jets = cms.InputTag("TCTauJetPlusTrackZSPCorJetAntiKt5")

JPTAntiKt5JetExtender = ak4JetExtender.clone()
JPTAntiKt5JetExtender.jets = cms.InputTag("ak4CaloJets")
JPTAntiKt5JetExtender.jet2TracksAtCALO = cms.InputTag("JPTAntiKt5JetTracksAssociatorAtCaloFace")
JPTAntiKt5JetExtender.jet2TracksAtVX = cms.InputTag("JPTAntiKt5JetTracksAssociatorAtVertex")


jptRecoTauProducer = cms.Sequence(
        JPTeidTight *
	TCTauJetPlusTrackZSPCorJetAntiKt5 *
        JPTAntiKt5JetTracksAssociatorAtVertex *
        caloRecoTauTagInfoProducer *
        JPTCaloRecoTauProducer
)

