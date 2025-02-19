import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRZT210_V1::All' 
process.prefer("GlobalTag")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

## process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
#
##process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

process.load("RecoMuon.MuonIdentification.muonIdProducerSequence_cff")
process.load("RecoMuon.MuonIdentification.links_cfi")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('MuonIdentification','TrackAssociator'),
    cout = cms.untracked.PSet(
        # threshold = cms.untracked.string('INFO')
        threshold = cms.untracked.string('DEBUG'),
	# noTimeStamps = cms.untracked.bool(True),
	# noLineBreaks = cms.untracked.bool(True)
	DEBUG = cms.untracked.PSet(
           limit = cms.untracked.int32(0)
	   ),
	MuonIdentification = cms.untracked.PSet(
	   limit = cms.untracked.int32(-1)
	),
	TrackAssociator = cms.untracked.PSet(
	   limit = cms.untracked.int32(-1)
	),
    ),
    debugModules = cms.untracked.vstring("muons")
)

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
options = cms.untracked.PSet(
    Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.source = cms.Source("PoolSource",
    fileNames =
cms.untracked.vstring(
         '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_V4P_TrackerPointing_v1/0000/04470A48-2081-DD11-8ECA-001A92810AC8.root'
	 
)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/muonid_1.24.root'),
    outputCommands = cms.untracked.vstring("drop *",
        "keep *_genParticleCandidates_*_*",
	"keep recoTracks_*_*_*",
	"keep recoTrackExtras_*_*_*",
	"keep recoMuons_*_*_*",
	"keep *_cscSegments_*_*",
	"keep *_dt4DSegments_*_*",
	"keep *_towerMaker_*_*",
	"keep *_*_*_TEST")
)

process.p = cms.Path(process.dtlocalreco*process.globalMuonLinks*process.muons)

process.muons.inputCollectionLabels = cms.VInputTag(cms.InputTag("ctfWithMaterialTracksP5"), 
						    cms.InputTag("globalMuonLinks"), 
						    cms.InputTag("cosmicMuons"))
process.muons.inputCollectionTypes = cms.vstring('inner tracks', 
						 'links', 
						 'outer tracks')

process.muons.fillIsolation = False
process.muons.minPt = 0.
process.muons.minP = 0.

process.e = cms.EndPath(process.out)
