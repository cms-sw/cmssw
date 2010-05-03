import FWCore.ParameterSet.Config as cms

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *
JPTZSPCorrectorICone5.ResponseMap   = 'CondFormats/JetMETObjects/data/CMSSW_31X_resptowers.txt'
JPTZSPCorrectorICone5.UseEfficiency = cms.bool(False)

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *
JetPlusTrackZSPCorJetAntiKt5.UseZSP = cms.bool(False)

from Configuration.StandardSequences.Reconstruction_cff import *
from RecoTauTag.RecoTau.CaloRecoTauTagInfoProducer_cfi import *
caloRecoTauTagInfoProducer.CaloJetTracksAssociatorProducer = cms.InputTag('JPTAntiKt5JetTracksAssociatorAtVertex')

jptRecoTauProducer = cms.Sequence(
	JetPlusTrackCorrectionsAntiKt5 *
        caloRecoTauTagInfoProducer *
        caloRecoTauProducer
)
