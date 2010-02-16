import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetPlusTrackCorrections_cfi import *
#JPTZSPCorrectorICone5.ResponseMap =   'JetMETCorrections/Configuration/data/CMSSW_31X_resptowers.txt'
#JPTZSPCorrectorICone5.EfficiencyMap = 'JetMETCorrections/Configuration/data/CMSSW_167_TrackNonEff_one.txt'
#JPTZSPCorrectorICone5.LeakageMap =    'JetMETCorrections/Configuration/data/CMSSW_167_TrackLeakage_one.txt'
#For CMSSW_3_1_6:
JPTZSPCorrectorICone5.NonEfficiencyFile     = 'CMSSW_167_TrackNonEff_one'
JPTZSPCorrectorICone5.NonEfficiencyFileResp = 'CMSSW_167_TrackLeakage_one'
JPTZSPCorrectorICone5.ResponseFile          = 'CMSSW_31X_resptowers'

from JetMETCorrections.Configuration.JetPlusTrackCorrections_cff import *
#from JetMETCorrections.Configuration.ZSPJetCorrections332_cff import *
from JetMETCorrections.Configuration.ZSPJetCorrections219_cff import *
JPTZSPCorrectorICone5.ResponseMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_31X_resptowers.txt")
JPTZSPCorrectorICone5.EfficiencyMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackNonEff_one.txt")
JPTZSPCorrectorICone5.LeakageMap = cms.string("JetMETCorrections/Configuration/data/CMSSW_167_TrackLeakage_one.txt")
ZSPJetCorJetIcone5.src = cms.InputTag("iterativeCone5CaloJets")

from Configuration.StandardSequences.Reconstruction_cff import *
from RecoTauTag.RecoTau.CaloRecoTauTagInfoProducer_cfi import *
caloRecoTauTagInfoProducer.CaloJetTracksAssociatorProducer = cms.InputTag('ZSPiterativeCone5JetTracksAssociatorAtVertex')

jptRecoTauProducer = cms.Sequence(
#        ZSPJetCorrectionsIcone5 *
#        JetPlusTrackCorrectionsIcone5 *
# in order to work with CMSSW_3_1_6: (to be deprecated)
	ZSPJetCorrections *
	JetPlusTrackCorrections *
        caloRecoTauTagInfoProducer *
        caloRecoTauProducer
)
