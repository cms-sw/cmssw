import FWCore.ParameterSet.Config as cms

process = cms.Process("PAT")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default          = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    PATSummaryTables = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
        #'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/TauolaTTbar-Summer08_IDEAL_V9_v1-AODSIM.80.root'
        '/store/relval/CMSSW_2_2_7/RelValWM/GEN-SIM-RECO/STARTUP_V9_v1/0004/1E84F77B-341C-DE11-8A99-0019DB29C5FC.root',
        '/store/relval/CMSSW_2_2_7/RelValWM/GEN-SIM-RECO/STARTUP_V9_v1/0004/34267FD6-1C1C-DE11-A836-001617C3B78C.root',
        '/store/relval/CMSSW_2_2_7/RelValWM/GEN-SIM-RECO/STARTUP_V9_v1/0004/68BF59CF-1C1C-DE11-AFA9-000423D98BC4.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')
process.GlobalTag.globaltag = cms.string('STARTUP_V9::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# PAT Layer 0+1
process.load("PhysicsTools.PatAlgos.patSequences_cff")

process.load("MuonAnalysis.MuonAssociators.muonL1Match_cfi")
process.muonL1Match.preselection = cms.string("")

process.allLayer1Muons.trigPrimMatch = cms.VInputTag(
    cms.InputTag("muonL1Match"),
    cms.InputTag("muonL1Match","propagatedReco"),
)

## Put your EDAnalyzer here
##   process.plots = cms.EDFilter("DataPlotter",
##       muons   = cms.InputTag("cleanLayer1Muons"),
##       muonCut = cms.string("")
##   )

process.p = cms.Path(
    process.muonL1Match *
    process.patDefaultSequence 
# * process.plots
)

process.TFileService = cms.Service("TFileService", 
    fileName = cms.string("plots.root")
)

