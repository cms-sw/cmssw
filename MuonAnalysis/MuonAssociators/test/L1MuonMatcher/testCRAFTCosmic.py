import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")

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
### Alternatively, you can make your own reco::Muons...
#process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi") # to get the muon mass
#process.lhcBasicMuons    = cms.EDProducer("ConcreteChargedCandidateProducer", src = cms.InputTag("..."), particleType = cms.string('mu+'))
#process.lhcBasicMuonsUAV = process.lhcBasicMuons.clone(src = cms.InputTag("lhcStandAloneMuonsBarrelOnly", "UpdatedAtVtx"))
### It would be more correct to use 'ConcreteStandAloneMuonCandidateProducer' (for which the track appears as 'muon' instead of 'tracker',
### but there are no dictionaries for it (see comment in testCRAFT.py); anyway, when using 'cosmicPropagationHypothesis' it doesn't matter

# Make l1extraParticles for the three bunch crossing L1A=0,+1,-1
from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import l1extraParticles
process.l1muonsAnyBX = l1extraParticles.clone(
    produceCaloParticles = False, ### we don't have digis for these
    centralBxOnly = False         ### this is the important point
)

# Trigger matching with a configuration more suitable for CRAFT
process.load("MuonAnalysis.MuonAssociators.muonL1Match_cfi")
process.muonL1Match.src     = cms.InputTag('muons') 
process.muonL1Match.matched = cms.InputTag('l1muonsAnyBX')
process.muonL1Match.useTrack = 'muon' # note: if you use 'ConcreteChargedCandidateProducer' this must be 'track'
process.muonL1Match.useState = 'innermost'
process.muonL1Match.maxDeltaPhi = 50.0*3.14/180.0 # 50 degrees, slide 3 of http://indico.cern.ch/getFile.py/access?contribId=2&resId=0&materialId=slides&confId=57817
process.muonL1Match.maxDeltaR   = 9999.0          # no deltaR match
process.muonL1Match.preselection = ""
process.muonL1Match.writeExtraInfo = True
process.muonL1Match.cosmicPropagationHypothesis = cms.bool(True) # must specify the type, as I'm adding a new parameter and not replacing one

####  Merge matches into pat::GenericParticle
## from PhysicsTools.PatAlgos.producersLayer1.genericParticleProducer_cfi import allLayer1GenericParticles 
## process.myMuons = allLayer1GenericParticles.clone(
##         src = "lhcBasicMuons", ...
####  Merge matches into pat::Muon
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import allLayer1Muons 
process.myMuons = allLayer1Muons.clone(
    muonSource = 'muons', # src = 'lhcBasicMuons'
    embedStandAloneMuon = True,
    addTrigMatch = True,
    isolation = cms.PSet(), isoDeposits = cms.PSet(), addGenMatch = False, addTeVRefits = False ## turn off unwanted pat::Muon features
)
process.myMuons.trigPrimMatch = cms.VInputTag( cms.InputTag("muonL1Match"), cms.InputTag("muonL1Match","propagatedReco") )
process.myMuons.userData.userInts.src = cms.VInputTag( cms.InputTag("muonL1Match", "bx"), cms.InputTag("muonL1Match", "quality"), cms.InputTag("muonL1Match", "isolated") )
process.myMuons.userData.userFloats.src = cms.VInputTag( cms.InputTag("muonL1Match", "deltaR") )

process.p = cms.Path(
    process.l1muonsAnyBX *  
    process.muonL1Match  *
    process.myMuons      
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('l1CosmicMuonsCRAFT.root'),
    outputCommands = cms.untracked.vstring(
        "drop *", 
        "keep *_myMuons__*",
        "keep *_ctfWithMaterialTracksP5LHCNavigation_*_*", ## keep also the tracks (and especially the TrackExtras, which can't be embedded in pat::Muon)
        "keep *_cosmicMuons_*_*",
        "keep *_globalCosmicMuons_*_*",
    )
)
process.end = cms.EndPath(process.out)

