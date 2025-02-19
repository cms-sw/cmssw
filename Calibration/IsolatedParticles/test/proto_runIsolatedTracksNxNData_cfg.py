import FWCore.ParameterSet.Config as cms

process = cms.Process("L1SKIM")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000000
process.MessageLogger.categories.append('L1GtTrigReport')
process.MessageLogger.categories.append('HLTrigReport')

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",fileNames =cms.untracked.vstring(
"/store/data/Run2010A/MinimumBias/RECO/Apr21ReReco-v1/0000/08275F4A-5270-E011-9DC3-003048635E02.root",
"/store/data/Run2010A/MinimumBias/RECO/Apr21ReReco-v1/0000/08042520-0A6D-E011-AECB-00304866C674.root",
"/store/data/Run2010A/MinimumBias/RECO/Apr21ReReco-v1/0000/06E42A32-0A6D-E011-87D0-003048673EBA.root",
"/store/data/Run2010A/MinimumBias/RECO/Apr21ReReco-v1/0000/06BD5531-756D-E011-AF1D-003048674096.root",
"/store/data/Run2010A/MinimumBias/RECO/Apr21ReReco-v1/0000/067F2E52-716F-E011-9738-0015170AD178.root",
"/store/data/Run2010A/MinimumBias/RECO/Apr21ReReco-v1/0000/067E8544-086D-E011-8A80-001A6478A824.root",
))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )

##################### digi-2-raw plus L1 emulation #########################

process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')

#################### Conditions and L1 menu ################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_42_V13::All' # used for Apr21 run2010A & run2010B


#################################################################################################
process.load("Calibration.IsolatedParticles.isolatedTracksNxN_cfi")
process.isolatedTracksNxN.JetSource = 'ak5CaloJets'
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('IsolatedTracksNxNData.root')
                                   )
#=============================================================================
# select the  events with trigger type
# configure Technical Bits
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')
#process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')

# select on MinBias trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.myHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
process.myHLT.HLTPaths = cms.vstring("HLT_MinBiasBSC","HLT_L1Tech_BSC_minBias")
#process.myHLT.HLTPaths = cms.vstring("HLT_MinBiasBSC","HLT_L1_BscMinBiasOR_BptxPlusORMinus","HLT_L1Tech_BSC_minBias","HLT_L1_BPTX")
process.myHLT.throw    = cms.bool(False)
#process.myHLT.HLTPaths = cms.vstring("HLT_MinBiasBSC")

# filter out scrapping events
process.noScraping= cms.EDFilter("FilterOutScraping",
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn     = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack    = cms.untracked.uint32(10),
                                 thresh      = cms.untracked.double(0.25)
                                 )

process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                           vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                           minimumNDOF      = cms.uint32(4) ,
                                           maxAbsZ          = cms.double(25.0),
                                           maxd0            = cms.double(2.0)
                                           )


#=============================================================================
# define an EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
      HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT')
)

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")
#process.l1GtTrigReport.L1GtRecordInputTag = 'simGtDigis'
process.l1GtTrigReport.L1GtRecordInputTag = 'gtDigis'
process.l1GtTrigReport.PrintVerbosity = 1
#=============================================================================

process.p1 = cms.Path( process.primaryVertexFilter * process.hltLevel1GTSeed * process.noScraping * process.myHLT * process.isolatedTracksNxN )

#process.e = cms.EndPath(process.l1GtTrigReport + process.hltTrigReport)
#process.e = cms.EndPath(process.l1GtTrigReport)
