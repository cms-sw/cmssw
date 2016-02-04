import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration/StandardSequences/Geometry_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
# for Beam
#process.load("Configuration.StandardSequences.Reconstruction_cff")
# for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.load("Configuration/StandardSequences/RawToDigi_cff")
process.load("CalibMuon.Configuration.getCSCConditions_frontier_cff")

# if you want to look at the simDigis rather than the raw2digi digis in simulation
#process.csc2DRecHits.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
#process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

# specify the global tag to use..
# more info and a list of current tags can be found at
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
#process.GlobalTag.globaltag = 'IDEAL_31X::All'
process.GlobalTag.globaltag ='START37_V4::All'
#'MC_37Y_V4::All'
#'START38_V3::All'

#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.myProcess = cms.ESSource("PoolDBESSource",
#                                  CondDBSetup,
#                                  connect = cms.string("sqlite_file:mySqliteFile.db"),
#                                  toGet = cms.VPSet(cms.PSet(record = cms.string("CSCDBGainsRcd"),
#                                                             tag = cms.string("CSCDBGains_2010_mc")
#                                                             tag = cms.string("CSCDBGains_express")                                                             
#                                                             ))
#                                  )
#process.es_prefer_mybadWires = cms.ESPrefer("PoolDBESSource","myProcess")

#### to use local sqlite file
process.cscConditions.connect='sqlite_file:mySqliteFile.db'
process.cscConditions.toGet = cms.VPSet(
        cms.PSet(record = cms.string('CSCDBGainsRcd'),
                 tag = cms.string('CSCDBGains_ME42')),
        cms.PSet(record = cms.string('CSCDBNoiseMatrixRcd'),
                 tag = cms.string('CSCDBNoiseMatrix_ME42')),
        cms.PSet(record = cms.string('CSCDBCrosstalkRcd'),
                 tag = cms.string('CSCDBCrosstalk_ME42')),
        cms.PSet(record = cms.string('CSCDBPedestalsRcd'),
                 tag = cms.string('CSCDBPedestals_ME42'))
)

process.es_prefer_cscConditions = cms.ESPrefer("PoolDBESSource","cscConditions")



# points to CMSSW_3_1_0_pre2 single muon (Pt = 100) relval sample.  Sim data must contain
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
#        '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/746C9E4E-D932-DE11-B1E6-001617DBCF90.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/5C7FE942-1733-DE11-880D-001617C3B77C.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/32F159D3-D832-DE11-9A86-000423D98A44.root'


#         '/store/relval/CMSSW_3_8_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V2-v1/0001/80164129-DA7A-DF11-AEFD-00304879FA4C.root'
                                
#          '/store/relval/CMSSW_3_7_0/RelValSingleMuPt100/GEN-SIM-RECO/MC_37Y_V4-v1/0025/BCFF63C8-5E69-DF11-B141-003048678FC4.root'
          '/store/relval/CMSSW_3_7_0/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0025/3248FA54-5F69-DF11-A59E-0030486792BA.root'
)
)

# recommend at least 10k events (single Muon Simulation)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# This is needed to avoid using RPC rechits in STA muon production
#process.standAloneMuons.STATrajBuilderParameters.FilterParameters.EnableRPCMeasurement = cms.bool(False)
#process.standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableRPCMeasurement = cms.bool(False)
process.cosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.EnableRPCMeasurement = cms.untracked.bool(False)
#process.SETMuonSeed.SETTrajBuilderParameters.FilterParameters.EnableRPCMeasurement = cms.bool(False)

process.cscValidation = cms.EDAnalyzer("CSCValidation",
    # name of file which will contain output
    rootFileName = cms.untracked.string('valHists_sim.root'),
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

# for RECO or SIM  (if digis were not saved, make sure to set useDigis = False)
#process.p = cms.Path(process.cscValidation)
# to look at raw2digi digis + standard reconstruction (semi-hack, requires 2 file solution, mainly for looking at relvals)
###process.p = cms.Path(process.muonCSCDigis * process.cscValidation)
process.p = cms.Path( process.muonCSCDigis *process.csc2DRecHits * process.cscSegments* process.cscValidation)
# for RAW (Cosmics) if you want to look at Trigger and Standalone info
#process.p = cms.Path(process.gtDigis *
#                     process.muonCSCDigis * process.csc2DRecHits * process.cscSegments *
#                     process.offlineBeamSpot * process.CosmicMuonSeedEndCapsOnly*process.cosmicMuonsEndCapsOnly *
#                     process.cscValidation)
#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments *
#                     process.offlineBeamSpot * process.CosmicMuonSeedEndCapsOnly * process.cosmicMuonsEndCapsOnly *
#                     process.cscValidation)

