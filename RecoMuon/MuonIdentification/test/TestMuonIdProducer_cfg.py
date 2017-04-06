import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

# process.load("Configuration.StandardSequences.MagneticField_cff")
# process.load("Configuration.StandardSequences.Geometry_cff")
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.load("Configuration.StandardSequences.Reconstruction_cff")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
# process.GlobalTag.globaltag = 'MC_36Y_V3::All'

process.load("RecoMuon.MuonIdentification.links_cfi")
# process.globalMuonLinks.inputCollection = cms.InputTag("muons","","RECO")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    # categories = cms.untracked.vstring('MuonIdentification','TrackAssociator'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        # threshold = cms.untracked.string('DEBUG'),
	# noTimeStamps = cms.untracked.bool(True),
	# noLineBreaks = cms.untracked.bool(True)
	DEBUG = cms.untracked.PSet(
           limit = cms.untracked.int32(0)
	   ),
	#MuonIdentification = cms.untracked.PSet(
	#   limit = cms.untracked.int32(-1)
	#),
	#TrackAssociator = cms.untracked.PSet(
	#   limit = cms.untracked.int32(-1)
	#),
    ),
    debugModules = cms.untracked.vstring("muons1stStes")
)

# import FWCore.Framework.test.cmsExceptionsFatalOption_cff
# options = cms.untracked.PSet(
#     Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
# )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_1_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2018_realistic_v17-v1/00000/0CD81D1D-9715-E711-8B2D-0CC47A7452D8.root'),
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('muonid.root'),
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

# Recreate links
process.muonid_study = cms.Path(process.globalMuonLinks* process.muonIdProducerSequence)

process.muons1stStep.inputCollectionLabels = cms.VInputTag(cms.InputTag("generalTracks"), 
                                                           cms.InputTag("globalMuonLinks"), 
                                                           cms.InputTag("standAloneMuons","UpdatedAtVtx"),
                                                           cms.InputTag("tevMuons","firstHit"),cms.InputTag("tevMuons","picky"),cms.InputTag("tevMuons","dyt"))
process.muons1stStep.inputCollectionTypes = cms.vstring('inner tracks', 
                                                        'links', 
                                                        'outer tracks',
                                                        'tev firstHit',
                                                        'tev picky',
                                                        'tev dyt')

# process.muons.fillIsolation = False
# process.muons.minPt = 0.
# process.muons.minP = 0.

process.e = cms.EndPath(process.out)
