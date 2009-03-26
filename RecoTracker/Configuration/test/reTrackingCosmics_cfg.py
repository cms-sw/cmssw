import FWCore.ParameterSet.Config as cms

process = cms.Process("ReTracking")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
process.load("Configuration.EventContent.EventContentCosmics_cff")

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT_ALL_V9::All"
process.prefer("GlobalTag")


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225_ReReco_FromTrackerPointing_v1/0003/141E2DC2-5FF9-DD11-8671-0019B9E714CE.root'
   )    
)

# output module
process.EVT = cms.OutputModule("PoolOutputModule",
    process.RecoTrackerFEVT,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RecoTrackerP5')),
    fileName = cms.untracked.string('reTrackingFromRECO_P5.root')
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## default is false


#Paths
process.allPath = cms.Path(process.siStripMatchedRecHits*process.siPixelRecHits*process.ctftracksP5) #produce only CTF tracks
#process.allPath = cms.Path(process.siStripMatchedRecHits*process.siPixelRecHits*process.rstracksP5) #produce only RS tracks
#process.allPath = cms.Path(process.siStripMatchedRecHits*process.siPixelRecHits*process.cosmictracksP5) #produce only CosmicTF tracks
#process.allPath = cms.Path(process.siStripMatchedRecHits*process.siPixelRecHits*process.tracksP5) #produce all track collection for cosmics
process.outpath = cms.EndPath(process.EVT)


############ custom configurations for special studies #############
#process.ckfTrackCandidatesP5.useHitsSplitting = cms.bool(False)  #avoid to split matched-hits for CTF tracks 
####################################################################
