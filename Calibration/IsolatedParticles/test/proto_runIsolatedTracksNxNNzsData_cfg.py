import FWCore.ParameterSet.Config as cms

process = cms.Process("L1SKIM")

process.load("Calibration.IsolatedParticles.isolatedTracksNxN_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100000
#process.MessageLogger.categories.append('L1GtTrigReport')
#process.MessageLogger.categories.append('HLTrigReport')

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

####################### configure pool source #############################
process.source = cms.Source("PoolSource",
 fileNames =cms.untracked.vstring(
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F8EDA42C-9BC6-DF11-8CAD-00261894385A.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F8D7F0A3-98C6-DF11-9642-0018F3D0965A.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F8D23680-94C6-DF11-91AC-0018F3D096DC.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F8C6D89F-98C6-DF11-92B3-002618943810.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F8BF2E5A-94C6-DF11-9A0C-002618943927.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F89C8FD4-9AC6-DF11-A55B-001A92971B88.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F895BB8B-9BC6-DF11-BE04-001A92971B68.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F888160F-93C6-DF11-B46F-0018F3D09686.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F87A30AD-93C6-DF11-B063-0018F3D096DC.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F823F82F-9BC6-DF11-97A7-0026189438F3.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F8146F8C-9AC6-DF11-B576-0018F3D09710.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F6FF8D8D-9BC6-DF11-863A-0018F3D09620.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F6E06F83-97C6-DF11-8C54-001A92971BD6.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F6B89D88-9BC6-DF11-930D-003048678EE2.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F67DE8AE-97C6-DF11-A08E-0018F3D09706.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/F6788CAE-99C6-DF11-9D50-001BFCDBD19E.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/FEE9AE8E-9BC6-DF11-95A4-00261894393C.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/FEC5A32D-9AC6-DF11-82E5-001A92971BD6.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/FE9446FA-97C6-DF11-98ED-003048679080.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/FE73FD57-9FC6-DF11-9657-0018F3D096DA.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/FE285975-94C6-DF11-AB0F-0018F3D096D2.root',
    '/store/data/Run2010A/HcalNZS/RECO/Sep17ReReco_v2/0019/FE0C5E28-99C6-DF11-8605-00261894385A.root'

##########run2010B Dec22Rereco
#    '/store/data/Run2010B/HcalNZS/RECO/Dec22ReReco_v1/0156/F8172904-4528-E011-BF6A-001A92971B7E.root',
#    '/store/data/Run2010B/HcalNZS/RECO/Dec22ReReco_v1/0156/F6D9B64A-4C28-E011-B75C-0018F3D096C8.root',
#    '/store/data/Run2010B/HcalNZS/RECO/Dec22ReReco_v1/0156/F6CB08D7-4928-E011-94D6-001BFCDBD1BE.root',
#    '/store/data/Run2010B/HcalNZS/RECO/Dec22ReReco_v1/0156/F6C84402-4A28-E011-B98E-00261894389E.root',
#    '/store/data/Run2010B/HcalNZS/RECO/Dec22ReReco_v1/0156/F6AB82C5-4728-E011-AC61-001A92810AD2.root',
#    '/store/data/Run2010B/HcalNZS/RECO/Dec22ReReco_v1/0156/F68CFD2D-4228-E011-8237-001A92971B7E.root'#    "/store/data/Run2010A/HcalNZS/RECO/v4/000/139/103/2634CE90-0385-DF11-B9B3-0030487CD17C.root"
    ),
 skipEvents = cms.untracked.uint32(0)
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50000) )

##################### digi-2-raw plus L1 emulation #########################
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
#process.load('TrackingTools/TrackAssociator/DetIdAssociatorESProducer_cff')

#################### Conditions and L1 menu ################################

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_311_V2::All'   # used for May27 ReReco

############ Skim the events according to the L1 seeds  ####################

#select on HLT_HcalNZS_8E29 trigger 
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
process.skimL1Seeds = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
process.skimL1Seeds.L1GtReadoutRecordTag = cms.InputTag("gtDigis")
process.skimL1Seeds.L1GtObjectMapTag     = cms.InputTag("hltL1GtObjectMap")
process.skimL1Seeds.L1CollectionsTag     = cms.InputTag("l1extraParticles")
process.skimL1Seeds.L1MuonCollectionTag  = cms.InputTag("l1extraParticles")
process.skimL1Seeds.L1SeedsLogicalExpression = "L1_SingleEG2 OR L1_SingleEG5 OR L1_SingleEG8 OR L1_SingleEG10 OR L1_SingleEG12 OR L1_SingleEG15 OR L1_SingleEG20 OR L1_SingleIsoEG5 OR L1_SingleIsoEG8 OR L1_SingleIsoEG10 OR L1_SingleIsoEG12 OR L1_SingleIsoEG15 OR L1_SingleJet6U OR L1_SingleJet10U OR L1_SingleJet20U OR L1_SingleJet30U OR L1_SingleJet40U OR L1_SingleJet50U OR L1_SingleJet60U OR L1_SingleTauJet10U OR L1_SingleTauJet20U OR L1_SingleTauJet30U OR L1_SingleTauJet50U OR L1_SingleMuOpen OR L1_SingleMu0 OR L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7 OR L1_SingleMu10 OR L1_SingleMu14 OR L1_SingleMu20 OR L1_ZeroBias"

# select on HLT_HcalPhiSym trigger
process.load("HLTrigger.HLTfilters.hltLevel1Activity_cfi")
process.hltLevel1Activity.L1GtReadoutRecordTag  = cms.InputTag('gtDigis')

######################## Configure Analyzer ###############################
#process.isotracksL1 = cms.EDAnalyzer("IsolatedTracksNxN",
#                                   DoMC                  = cms.untracked.bool(False),
#                                   WriteAllTracks        = cms.untracked.bool(False),
#                                     Verbosity             = cms.untracked.int32( 1 ),
#                                   PVTracksPtMin         = cms.untracked.double( 0.200 ),
#                                   DebugTracks           = cms.untracked.int32(0),
#                                   PrintTrkHitPattern    = cms.untracked.bool(True),
#                                   minTrackP             = cms.untracked.double( 1.0 ),
#                                   maxTrackEta           = cms.untracked.double( 2.6 ),
#                                   DebugL1Info           = cms.untracked.bool(False),
#                                   L1TriggerAlgoInfo     = cms.untracked.bool(False),
#                                   L1extraTauJetSource   = cms.InputTag("l1extraParticles", "Tau"),
#                                   L1extraCenJetSource   = cms.InputTag("l1extraParticles", "Central"),
#                                   L1extraFwdJetSource   = cms.InputTag("l1extraParticles", "Forward"),
#                                   L1extraMuonSource     = cms.InputTag("l1extraParticles"),
#                                   L1extraIsoEmSource    = cms.InputTag("l1extraParticles","Isolated"),
#                                   L1extraNonIsoEmSource = cms.InputTag("l1extraParticles","NonIsolated"),
#                                   L1GTReadoutRcdSource  = cms.InputTag("gtDigis"),
#                                   L1GTObjectMapRcdSource= cms.InputTag("hltL1GtObjectMap"),
#                                   JetSource             = cms.InputTag("iterativeCone5CaloJets"),
#                                   JetExtender           = cms.InputTag("iterativeCone5JetExtender"),
#                                     HBHERecHitSource      = cms.InputTag("hbherecoMB"),
#                                   maxNearTrackPT        = cms.untracked.double( 1.0 ),
#                                   TimeMinCutECAL        = cms.untracked.double(-500.0),
#                                   TimeMaxCutECAL        = cms.untracked.double(500.0),
#                                   TimeMinCutHCAL        = cms.untracked.double(-500.0),
#                                   TimeMaxCutHCAL        = cms.untracked.double(500.0),
#                                   DebugEcalSimInfo      = cms.untracked.int32(2),
#                                  )

process.isolatedTracksNxN.Verbosity = cms.untracked.int32( 0 )
process.isolatedTracksNxN.HBHERecHitSource = cms.InputTag("hbhereco")
process.isolatedTracksNxN.L1TriggerAlgoInfo = True
#process.isolatedTracksNxN.DebugL1Info = True

process.isolatedTracksNxN_NZS = process.isolatedTracksNxN.clone(
    Verbosity = cms.untracked.int32( 0 ),
    HBHERecHitSource = cms.InputTag("hbherecoMB"),
    L1TriggerAlgoInfo = True
    )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('IsolatedTracksNxNData.root')
                                   )

# configure Technical Bits to ensure collision and remove BeamHalo
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND NOT (36 OR 37 OR 38 OR 39)')

# filter out scrapping events
process.noScraping= cms.EDFilter("FilterOutScraping",
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn     = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack    = cms.untracked.uint32(10),
                                 thresh      = cms.untracked.double(0.25)
                                 )

# select on primary vertex
process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                           vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                           minimumNDOF      = cms.uint32(4) ,
                                           maxAbsZ          = cms.double(25.0),
                                           maxd0            = cms.double(5.0)
                                           )


#=============================================================================
# define an EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
      HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT')
)

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
process.l1GtTrigReport.L1GtRecordInputTag = 'gtDigis'
process.l1GtTrigReport.PrintVerbosity = 1
#=============================================================================

#### by Benedikt
process.p1 = cms.Path(process.primaryVertexFilter * process.hltLevel1GTSeed * process.noScraping * process.skimL1Seeds       *process.isolatedTracksNxN * process.isolatedTracksNxN_NZS)
#process.p1 = cms.Path( process.hltLevel1GTSeed * process.noScraping  *process.isolatedTracksNxN * process.isolatedTracksNxN_NZS)
process.e = cms.EndPath(process.l1GtTrigReport + process.hltTrigReport)
#process.e = cms.EndPath(process.hltTrigReport)
