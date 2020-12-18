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

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0015/3014AE2E-6503-DE11-B093-003048767DCD.root',
        '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0012/EA27ED04-0602-DE11-B31E-001A92971B8C.root'
    ),
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTag.globaltag = cms.string('CRAFT_ALL_V9::All')

from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import l1extraParticles
process.l1muonsAnyBX = l1extraParticles.clone(
    #muonSource = cms.InputTag( "hltGtDigis" ),
    produceCaloParticles = False, ### we don't have digis for these
    centralBxOnly = False         ### this is the important point
)

### one could also convert STA track to a Candidate instead of using the reco::Muon, 
### but one needs to add the dictionary for std::vector<reco::RecoStandAloneMuonCandidate> and it's edm::Wrapper
### in DataFormats/RecoCandidate. 
### Alternatively, you can make your own reco::Muons...
#process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi") # to get the muon mass
#process.lhcBasicMuons    = cms.EDProducer("ConcreteStandAloneMuonCandidateProducer", src = cms.InputTag("lhcStandAloneMuonsBarrelOnly"), particleType = cms.string('mu+'))
#process.lhcBasicMuonsUAV = process.lhcBasicMuons.clone(src = cms.InputTag("lhcStandAloneMuonsBarrelOnly", "UpdatedAtVtx"))

# Make l1extraParticles for the three bunch crossing L1A=0,+1,-1
from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import l1extraParticles
process.l1muonsAnyBX = l1extraParticles.clone(
    produceCaloParticles = False, ### we don't have digis for these
    centralBxOnly = False         ### this is the important point
)

# Trigger matching with a configuration more suitable for CRAFT
process.load("MuonAnalysis.MuonAssociators.muonL1Match_cfi")
process.muonL1Match.src     = cms.InputTag('lhcSTAMuonsBarrelOnly') ## cms.InputTag("lhcBasicMuons")
process.muonL1Match.matched = cms.InputTag('l1muonsAnyBX')
process.muonL1Match.useTrack = 'muon'
process.muonL1Match.useState = 'innermost'
process.muonL1Match.maxDeltaPhi = 50.0*3.14/180.0 # 50 degrees, slide 3 of http://indico.cern.ch/getFile.py/access?contribId=2&resId=0&materialId=slides&confId=57817
process.muonL1Match.maxDeltaR   = 9999.0          # no deltaR match
process.muonL1Match.preselection = ""
process.muonL1Match.writeExtraInfo = True
#process.muonL1MatchUAV = process.muonL1Match.clone(src = 'lhcBasicMuonsUAV')

####  Merge matches into pat::GenericParticle
## from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import allLayer1GenericParticles 
## process.lhcMuons = allLayer1GenericParticles.clone(
##         src = "lhcBasicMuons", ...
####  Merge matches into pat::Muon
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import allLayer1Muons 
process.lhcMuons = allLayer1Muons.clone(
    muonSource = 'lhcSTAMuonsBarrelOnly', # src = 'lhcBasicMuons'
    embedStandAloneMuon = True,
    addTrigMatch = True,
    isolation = cms.PSet(), isoDeposits = cms.PSet(), addGenMatch = False, addTeVRefits = False ## turn off unwanted pat::Muon features
)
#process.lhcMuonsUAV = process.lhcMuons.clone(src = 'lhcBasicMuonsUAV')

def addMatch(muons, matcher):
    muons.trigPrimMatch = cms.VInputTag( cms.InputTag(matcher), cms.InputTag(matcher,"propagatedReco") )
    muons.userData.userInts.src = cms.VInputTag( cms.InputTag(matcher, "bx"), cms.InputTag(matcher, "quality"), cms.InputTag(matcher, "isolated") )
    muons.userData.userFloats.src = cms.VInputTag( cms.InputTag(matcher, "deltaR") )
addMatch(process.lhcMuons, "muonL1Match")
#addMatch(process.lhcMuonsUAV, "muonL1MatchUAV")

process.p = cms.Path(
    process.l1muonsAnyBX *  
    #process.lhcBasicMuons * process.lhcBasicMuonsUAV *
    process.muonL1Match   *# process.muonL1MatchUAV *
    process.lhcMuons      #* process.lhcMuonsUAV
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('l1muonsCRAFT.root'),
    outputCommands = cms.untracked.vstring("drop *", "keep *_lhcMuons__*"), #, "keep *_lhcMuonsUAV__*"),
)
process.end = cms.EndPath(process.out)

