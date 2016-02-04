#
# This example configuration file dumps FFTJet energy discretization grids
# in the form of root histograms.
#
# I. Volobouev, June 29 2011
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("FFTJetTest")

# Various standard stuff
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'FT_R_42_V13A::All'

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("fftjet_grids.root")
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/FA15E8AA-F782-E011-9D3C-001BFCDBD1BE.root'
    )
)

# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring(
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/FE8F09A1-8082-E011-990F-001A92971B7E.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/FA15E8AA-F782-E011-9D3C-001BFCDBD1BE.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/F88FDE18-6782-E011-B5D8-003048678ED4.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/F839BD95-F183-E011-BE30-002618FDA262.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/D89DE790-FC82-E011-8CA6-0026189438CB.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/C4BE9561-6782-E011-A791-0030486792B4.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/C0CE5857-8482-E011-9BD9-002354EF3BDE.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/BC5BF6CB-6682-E011-B5E2-00248C0BE01E.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/AA6C29E6-6F82-E011-8F12-001A92971B62.root',
#         '/store/data/Run2011A/MinimumBias/RECO/May10ReReco-v2/0002/A476C008-8A82-E011-A676-002354EF3BDA.root'
#     )
# )

# Configure the trigger
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed

# L1 technical triggers:
# 0     -- beam crossing
# 36-39 -- beam halo
process.VetoHardInt = hltLevel1GTSeed.clone(
    L1TechTriggerSeeding = cms.bool(True),
    L1SeedsLogicalExpression = cms.string('0 AND NOT (36 OR 37 OR 38 OR 39)')
) 

# Remove the so-called "scraping" events
process.noScraping = cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
)

# min bias filter
process.HLTZeroBias =cms.EDFilter("HLTHighLevel",
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    HLTPaths = cms.vstring('HLT_ZeroBias','HLT_ZeroBias_v1','HLT_ZeroBias_v2','HLT_ZeroBias_v3'),
    eventSetupPathsKey = cms.string(''),
    andOr = cms.bool(True), #----- True = OR, False = AND between the HLTPaths
    throw = cms.bool(False)
)

# Configure the FFTJet pattern recognition module
from RecoJets.FFTJetProducers.fftjetcommon_cfi import *
from RecoJets.FFTJetProducers.fftjetpatrecoproducer_cfi import *

fftjet_patreco_producer.src = cms.InputTag("particleFlow")
fftjet_patreco_producer.jetType = cms.string("PFJet")
fftjet_patreco_producer.storeDiscretizationGrid = cms.bool(True)
fftjet_patreco_producer.makeClusteringTree = cms.bool(False)
fftjet_patreco_producer.GridConfiguration = fftjet_grid_256_128

# Configure the FFTJet pile-up analyzer module to dump the grids
from RecoJets.JetAnalyzers.fftjetpileupanalyzer_cfi import *

fftjet_pileup_analyzer.collectPileup = cms.bool(False)
fftjet_pileup_analyzer.collectSummaries = cms.bool(False)
fftjet_pileup_analyzer.collectGrids = cms.bool(True)
fftjet_pileup_analyzer.collectVertexInfo = cms.bool(True)

process.fftjetpatreco = fftjet_patreco_producer
process.pileupanalyzer = fftjet_pileup_analyzer

process.p = cms.Path(
    process.VetoHardInt *
    process.HLTZeroBias * 
    process.noScraping * 
    process.fftjetpatreco *
    process.pileupanalyzer
)
