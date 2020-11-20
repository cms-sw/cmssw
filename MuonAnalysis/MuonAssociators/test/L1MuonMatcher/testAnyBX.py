import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")

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
        'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/TauolaTTbar-Summer08_IDEAL_V9_v1-AODSIM.80.root'
        #'/store/relval/CMSSW_2_2_7/RelValWM/GEN-SIM-RECO/STARTUP_V9_v1/0004/1E84F77B-341C-DE11-8A99-0019DB29C5FC.root',
        #'/store/relval/CMSSW_2_2_7/RelValWM/GEN-SIM-RECO/STARTUP_V9_v1/0004/34267FD6-1C1C-DE11-A836-001617C3B78C.root',
        #'/store/relval/CMSSW_2_2_7/RelValWM/GEN-SIM-RECO/STARTUP_V9_v1/0004/68BF59CF-1C1C-DE11-AFA9-000423D98BC4.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')
process.GlobalTag.globaltag = cms.string('STARTUP_V9::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# PAT Layer 0+1
process.load("PhysicsTools.PatAlgos.patSequences_cff")
#process.allLayer1Jets.jetCorrFactorsSource = cms.VInputTag(cms.InputTag("jetCorrFactors"))
#process.patJetMETCorrections.remove(process.jptJetCorrFactors)
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run22XonSummer08AODSIM
run22XonSummer08AODSIM(process)

from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import l1extraParticles
process.l1muonsAnyBX = l1extraParticles.clone(
    #muonSource = cms.InputTag( "hltGtDigis" ),
    produceCaloParticles = False, ### we don't have digis for these
    centralBxOnly = False         ### this is the important point
)

process.load("MuonAnalysis.MuonAssociators.muonL1Match_cfi")
process.muonL1Match.preselection = cms.string("")
process.muonL1Match.writeExtraInfo = cms.bool(True)
process.muonL1Match.matched = cms.InputTag('l1muonsAnyBX')

process.allLayer1Muons.trigPrimMatch = cms.VInputTag(
    cms.InputTag("muonL1Match"),
    cms.InputTag("muonL1Match","propagatedReco"),
)
process.allLayer1Muons.userData.userInts.src = cms.VInputTag(
    cms.InputTag("muonL1Match", "bx"),
    cms.InputTag("muonL1Match", "quality"),
    cms.InputTag("muonL1Match", "isolated"),
)
process.allLayer1Muons.userData.userFloats.src = cms.VInputTag(
    cms.InputTag("muonL1Match", "deltaR")
)
process.selectedLayer1Muons.cut = "!triggerMatchesByFilter('l1').empty()"
process.selectedLayer1Muons.filter = cms.bool(True)

process.p = cms.Path(
    process.l1muonsAnyBX *
    process.muonL1Match *
    process.patDefaultSequence 
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('l1muonsAnyBX.root'),
    outputCommands = cms.untracked.vstring("drop *", "keep *_cleanLayer1Muons__*"),
)
process.end = cms.EndPath(process.out)

