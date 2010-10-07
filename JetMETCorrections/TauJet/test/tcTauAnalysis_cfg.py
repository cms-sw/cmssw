import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("test")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'/store/relval/CMSSW_3_6_0_pre1/RelValZTT/GEN-SIM-RECO/START3X_V21-v1/0002/1E8AE923-2922-DF11-B460-0030487CD7B4.root'
"rfio:/castor/cern.ch/user/s/slehti/testData/Ztautau_Spring10-START3X_V26_S09-v1-RAW-RECO.root"
    )
)

process.load("FWCore/MessageService/MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')
process.GlobalTag.globaltag = cms.string('START36_V2::All')

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.load("RecoTauTag.Configuration.RecoTCTauTag_cff")
#process.tcRecoTauProducer.DropCaloJets = cms.untracked.bool(True)
#process.tcRecoTauProducer.DropRejectedJets = cms.untracked.bool(False)
process.load("JetMETCorrections.TauJet.TCRecoTauDiscriminationAgainstHadronicJets_cfi")
process.load("JetMETCorrections.TauJet.TCRecoTauDiscriminationAlgoComponent_cfi")

process.tcTauCorrectorTest = cms.EDAnalyzer("TCTauAnalysis",

	CaloTauCollection = cms.InputTag("caloRecoTauProducer"),
	PFTauCollection	  = cms.InputTag("shrinkingConePFTauProducer"),
	MCTauCollection   = cms.InputTag("TauMCProducer:HadronicTauOneAndThreeProng"),

	Discriminator	  = cms.InputTag("DiscriminationByIsolation"),

	TauJetEt          = cms.double(20.),
        TauJetEta         = cms.double(2.1),

	UseMCInfo         = cms.bool(True)
)

#TauMCProducer
process.load("HLTriggerOffline.Tau.Validation.HLTTauReferences_cfi")

process.runEDAna = cms.Path(
	process.TauMCProducer *
        process.tautagging *
	process.tcRecoTauDiscriminationAgainstHadronicJets *
	process.tcRecoTauDiscriminationAlgoComponent *
	process.tcTauCorrectorTest
)

#process.TESTOUT = cms.OutputModule("PoolOutputModule",
#        outputCommands = cms.untracked.vstring(
#            "keep *"
#        ),
#        fileName = cms.untracked.string('file:testout.root')
#)
#process.outpath = cms.EndPath(process.TESTOUT)
