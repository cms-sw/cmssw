import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("test")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'rfio:/castor/cern.ch/user/s/slehti/testData/Ztautau_GEN_SIM_RECO_MC_31X_V2_preproduction_311_v1.root'
    )
)

process.load("FWCore/MessageService/MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP31X_V1::All'
process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.load("JetMETCorrections/TauJet/TCTauProducer_cff")
#process.tcRecoTauProducer.DropCaloJets = cms.untracked.bool(True)

process.tcTauCorrectorTest = cms.EDAnalyzer("TCTauAnalysis",

	CaloTauCollection	= cms.InputTag("caloRecoTauProducer","","RECO"),
	TCTauCollection		= cms.InputTag("tcRecoTauProducer"),
	PFTauCollection		= cms.InputTag("shrinkingConePFTauProducer"),
	MCTauCollection         = cms.InputTag("TauMCProducer:HadronicTauOneAndThreeProng"),

	Discriminator		= cms.InputTag("DiscriminationByIsolation"),

	TauJetEt       = cms.double(20.),
        TauJetEta      = cms.double(2.1)
)

#TauMCProducer
process.load("HLTriggerOffline.Tau.Validation.HLTTauReferences_cfi")

process.runEDAna = cms.Path(
	process.TauMCProducer *
        process.TCTau *
	process.tcTauCorrectorTest
)

#process.TESTOUT = cms.OutputModule("PoolOutputModule",
#        outputCommands = cms.untracked.vstring(
#            "keep *"
#        ),
#        fileName = cms.untracked.string('file:testout.root')
#)
#process.outpath = cms.EndPath(process.TESTOUT)
