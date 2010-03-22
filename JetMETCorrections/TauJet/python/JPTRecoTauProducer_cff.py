import FWCore.ParameterSet.Config as cms

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cfi import *
JPTZSPCorrectorICone5.ResponseMap =   'CondFormats/JetMETObjects/data/CMSSW_31X_resptowers.txt'
JPTZSPCorrectorICone5.EfficiencyMap = 'CondFormats/JetMETObjects/data/CMSSW_167_TrackNonEff_one.txt'
JPTZSPCorrectorICone5.LeakageMap =    'CondFormats/JetMETObjects/data/CMSSW_167_TrackLeakage_one.txt'

from RecoJets.JetPlusTracks.JetPlusTrackCorrections_cff import *
#from RecoJets.JetPlusTracks.ZSPJetCorrections332_cff import *

#from JetMETCorrections.Configuration.ZSPJetCorrections219_cff import *
#JPTZSPCorrectorICone5.ResponseMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_31X_resptowers.txt")
#JPTZSPCorrectorICone5.EfficiencyMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackNonEff_one.txt")
#JPTZSPCorrectorICone5.LeakageMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackLeakage_one.txt")
#ZSPJetCorJetIcone5.src = cms.InputTag("iterativeCone5CaloJets")

from Configuration.StandardSequences.Reconstruction_cff import *
from RecoTauTag.RecoTau.CaloRecoTauTagInfoProducer_cfi import *
caloRecoTauTagInfoProducer.CaloJetTracksAssociatorProducer = cms.InputTag('JPTiterativeCone5JetTracksAssociatorAtVertex')

jptRecoTauProducer = cms.Sequence(
	JetPlusTrackCorrectionsIcone5 *
        caloRecoTauTagInfoProducer *
        caloRecoTauProducer
)
