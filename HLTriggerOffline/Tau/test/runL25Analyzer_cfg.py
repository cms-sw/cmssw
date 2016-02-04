import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("TauL25Analysis")


process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START38_V12::All'
process.load('CondCore.DBCommon.CondDBSetup_cfi')
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100 # print the event number for every 100th event
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(
    #threshold = cms.untracked.string("DEBUG"),    # pring LogDebugs and above
    #threshold = cms.untracked.string("INFO")     # print LogInfos and above
    threshold = cms.untracked.string("WARNING")  # print LogWarnings and above
)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
#process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")

process.maxEvents = cms.untracked.PSet( 
	input = cms.untracked.int32(-1) 
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/user/eluiggi/QCD_Pt-20_MuEnrichedPt-15_TuneZ2_7TeV-pythia6/TTEffMuSkimQCDMuEnrichedPt15/f7e731948547950cb9ef05e9314c343e/TTEffSkim_2_1_QO0.root'
    )
)

#Mike needs Calo Geometry
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
#process.load("HLTriggerOffline/Tau/HLT_cff")
process.load("ElectroWeakAnalysis.TauTriggerEfficiency.TTEffAnalysisHLT_cfi")
process.prefer("magfield")
process.hltGctDigis.hltMode = cms.bool(False) # Making L1CaloRegions

#process.load("HLTriggerOffline/Tau/pftauSkim_cff")
process.load("ElectroWeakAnalysis.TauTriggerEfficiency.TTEffPFTau_cff")

process.load("L1Trigger/Configuration/L1Config_cff")


#match L25 to openhltL2TauJets
process.TTEffAnalysis = cms.EDAnalyzer("L25TauAnalyzer",
        PFTauSource = cms.InputTag("MyPFTausSelected"),
	PFTauIsoSource = cms.InputTag("thisPFTauDiscriminationByIsolation"),
	PFTauMuonDiscSource = cms.InputTag("thisPFTauDiscriminationAgainstMuon"),
	PrimaryVtxSource = cms.InputTag("hltPixelVertices"),
        #PFTauSource = cms.InputTag("fixedConePFTauProducer"),
	L2TauSource = cms.InputTag("openhltL2TauIsolationProducer"),
        L25JetSource = cms.InputTag("openhltL25TauConeIsolation"),
        L2L25MatchingCone = cms.double(0.3),
	L25JetLeadTkMatchingCone = cms.double(0.2),
	MinTrackPt = cms.double(1.0),
	SignalCone = cms.double(0.15),
	L25LeadTkMinPt = cms.double(5.0),
	IsolationCone = cms.double(0.5),
	L25DzCut = cms.double(0.2),
	NTrkIso = cms.int32(0)
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
    process.TTEffPFTau *
    process.TTEffAnalysis
) 

process.TFileService = cms.Service("TFileService", 
	fileName = cms.string("l25Analyzer.root") ,
	closeFileFast = cms.untracked.bool(True)
)

process.schedule = cms.Schedule(
        process.DoHLTJetsU,
	process.DoHLTTau,
	process.runEDAna
)
