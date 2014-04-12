import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# Standard stuff needed for the unpacking and local reco
process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
# for Beam
#process.load("Configuration.StandardSequences.Reconstruction_cff")
# for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

# cscSkim if you want to use it
process.load("DPGAnalysis/Skims/CSCSkim_cfi");

# specify the global tag to use..
# more info and a list of current tags can be found at
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.GlobalTag.globaltag = 'GR_R_36X_V10::All'

# this points to CRAFT '08 data (r68100) as an example
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#    'rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_CSCSkim_trial_225-v3/0009/FEF9A8E4-D000-DE11-9945-001731AF698F.root'
#    '/store/data/Run2010A/MinimumBias/ALCARECO/v1/000/135/993/10592DCB-2065-DF11-9C55-000423D992DC.root'
    '/store/data/Run2010A/MinimumBias/RAW-RECO/MuonDPG_skim-May27thSkim_v3/0033/E64CD56F-286F-DF11-9F0A-00E081791737.root'
)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        FwkJob = cms.untracked.PSet( limit = cms.untracked.int32(0) )
    ),
    categories = cms.untracked.vstring('FwkJob'),
    destinations = cms.untracked.vstring('cout')
)


# recommend at least 100k events for normal Cosmic running (less if running over CSCSkim)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# This is needed to avoid using RPC rechits in STA muon production
#process.standAloneMuons.STATrajBuilderParameters.FilterParameters.EnableRPCMeasurement = cms.bool(False)
#process.standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableRPCMeasurement = cms.bool(False)
process.cosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.EnableRPCMeasurement = cms.untracked.bool(False)
#process.SETMuonSeed.SETTrajBuilderParameters.FilterParameters.EnableRPCMeasurement = cms.bool(False)

# This is the CSCValidation package minimum block.  There are more input variables which
# can be set.  Check src/CSCValidation.cc to see what they are.
process.cscValidation = cms.EDAnalyzer("CSCValidation",
    # name of file which will contain output
    rootFileName = cms.untracked.string('valHists_global.root'),
    # basically turns on/off residual plots which use simhits
    isSimulation = cms.untracked.bool(False),
    # stores a tree of info for the first 1.5M rechits and 2M segments
    # used to make 2D scatter plots of global positions.  Significantly increases
    # size of output root file, so beware...
    writeTreeToFile = cms.untracked.bool(True),
    # mostly for MC and RECO files which may have dropped the digis
    useDigis = cms.untracked.bool(False),
    # lots of extra, more detailed plots
    detailedAnalysis = cms.untracked.bool(False),
    # set to true to only look at events with CSC L1A
    useTriggerFilter = cms.untracked.bool(False),
    # set to true to only look at events with clean muon (based on STA muon)
    useQualityFilter = cms.untracked.bool(False),
    alctDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCALCTDigi"),
    clctDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCCLCTDigi"),
    corrlctDigiTag =  cms.InputTag("simMuonCSCDigis","MuonCSCCorrelatedLCTDigi"),                                       
    # Input tags for various collections CSCValidation looks at
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    compDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
    cscRecHitTag = cms.InputTag("csc2DRecHits"),
    cscSegTag = cms.InputTag("cscSegments"),
    saMuonTag = cms.InputTag("cosmicMuonsEndCapsOnly"),
    l1aTag = cms.InputTag("gtDigis"),
    simHitTag = cms.InputTag("g4SimHits", "MuonCSCHits")
)

# for RECO (if digis were not saved, make sure to set useDigis = False)
###process.p = cms.Path(process.cscSkim * process.cscValidation)
process.p = cms.Path( process.muonCSCDigis *process.csc2DRecHits * process.cscSegments* process.cscValidation)
# for RAW with just local level CSC Stuff
#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments *
#                     process.offlineBeamSpot * process.CosmicMuonSeedEndCapsOnly * process.cosmicMuonsEndCapsOnly *
#                     process.cscValidation)
# for RAW (Cosmics) if you want to look at Trigger 
#process.p = cms.Path(process.gtDigis *
#                     process.muonCSCDigis * process.csc2DRecHits * process.cscSegments *
#                     process.offlineBeamSpot * process.CosmicMuonSeedEndCapsOnly*process.cosmicMuonsEndCapsOnly *
#                     process.cscValidation)

