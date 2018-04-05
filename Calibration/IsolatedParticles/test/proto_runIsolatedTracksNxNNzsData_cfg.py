import FWCore.ParameterSet.Config as cms

process = cms.Process("L1SKIM")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100000

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

####################### configure pool source #############################
process.source = cms.Source("PoolSource",
 fileNames =cms.untracked.vstring(
    '/store/data/Run2010A/MinimumBias/RECO/Apr21ReReco-v1/0000/08275F4A-5270-E011-9DC3-003048635E02.root'
    ),
 skipEvents = cms.untracked.uint32(0)
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

##################### digi-2-raw plus L1 emulation #########################
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

#################### Conditions and L1 menu ################################

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_data']

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

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Calibration.IsolatedParticles.isolatedTracksNxN_cfi")
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
process.e = cms.EndPath(process.l1GtTrigReport + process.hltTrigReport)

