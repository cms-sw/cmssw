import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("test")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'file:/tmp/slehti/A161_RECO.root'
#	'/store/relval/CMSSW_3_1_0_pre11/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/A2CC8319-F064-DE11-B0C6-00304876A0FF.root'
	'rfio:/castor/cern.ch/user/s/slehti/testData/Ztautau_GEN_SIM_RECO_MC_31X_V2_preproduction_311_v1.root'
    )
)

process.load("FWCore/MessageService/MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_V1::All'
process.GlobalTag.globaltag = 'STARTUP31X_V1::All'

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
process.load("TrackingTools.TrackAssociator.default_cfi")

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")

process.tcTauCorrectorTest = cms.EDAnalyzer("TCTauAnalysis",
	## Optional input for the TCTauAlgorithm
        ## uncomment here any line which needs changing.
	# EtCaloOverTrackMin 	= cms.untracked.double(-0.9),
	# EtCaloOverTrackMax	= cms.untracked.double(0.0),
	# EtHcalOverTrackMin	= cms.untracked.double(-0.3),
	# EtHcalOverTrackMax	= cms.untracked.double(1.0),
	# SignalConeSize	= cms.untracked.double(0.2),
	# EcalConeSize		= cms.untracked.double(0.5),
	# MatchingConeSize	= cms.untracked.double(0.1),
	# Track_minPt		= cms.untracked.double(1.0),
	# tkmaxipt		= cms.untracked.double(0.03),
	# tkmaxChi2		= cms.untracked.double(100.),
	# tkminPixelHitsn	= cms.untracked.int32(2),
	# tkminTrackerHitsn	= cms.untracked.int32(8),
	# TrackCollection	= cms.untracked.InputTag("generalTracks"),
	# PVProducer		= cms.untracked.InputTag("offlinePrimaryVertices"),
	# EBRecHitCollection	= cms.untracked.InputTag("ecalRecHit:EcalRecHitsEB"),
	# EERecHitCollection	= cms.untracked.InputTag("ecalRecHit:EcalRecHitsEE"),
	# HBHERecHitCollection	= cms.untracked.InputTag("hbhereco"),
	# HORecHitCollection	= cms.untracked.InputTag("horeco"),
	# HFRecHitCollection	= cms.untracked.InputTag("hfreco"),
	TrackAssociatorParameters = process.TrackAssociatorParameterBlock.TrackAssociatorParameters,

	MCTauCollection         = cms.InputTag("TauMCProducer:HadronicTauOneAndThreeProng"),

        ## for the test program: ProngSelection = "1prong","3prong","any" (any = 1 or 3 prong)
#        ProngSelection = cms.string("1prong"),
        ## for the test program: TauJet jet energy correction parameters
        src            = cms.InputTag("iterativeCone5CaloJets"),
        tagName        = cms.string("IterativeCone0.4_EtScheme_TowerEt0.5_E0.8_Jets871_2x1033PU_tau"),
        TauTriggerType = cms.int32(1),
	TauJetEt       = cms.double(20.),
        TauJetEta      = cms.double(2.1)
)

process.TauMCProducer = cms.EDProducer("HLTTauMCProducer",
	GenParticles  = cms.untracked.InputTag("genParticles"),
       	ptMinTau      = cms.untracked.double(3),
       	ptMinMuon     = cms.untracked.double(3),
       	ptMinElectron = cms.untracked.double(3),
       	BosonID       = cms.untracked.vint32(23),
       	EtaMax         = cms.untracked.double(2.5)
)

process.runEDAna = cms.Path(
	process.TauMCProducer *
        process.ZSPJetCorrections *
        process.ZSPrecoJetAssociations *
        process.JetPlusTrackCorrections *
	process.tcTauCorrectorTest
)
