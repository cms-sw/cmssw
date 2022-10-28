import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

# process.load("Configuration.StandardSequences.GeometryDB_cff")

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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')
# process.GlobalTag.globaltag = 'MC_36Y_V3::All'

process.load("RecoMuon.MuonIdentification.links_cfi")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# process.MessageLogger = cms.Service("MessageLogger",
#     destinations = cms.untracked.vstring('cout'),
#     categories = cms.untracked.vstring('MuonIdentification','TrackAssociator'),
#     cout = cms.untracked.PSet(
#         threshold = cms.untracked.string('INFO'),
#         # threshold = cms.untracked.string('DEBUG'),
# 	# noTimeStamps = cms.untracked.bool(True),
# 	# noLineBreaks = cms.untracked.bool(True)
# 	DEBUG = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
# 	   ),
# 	#MuonIdentification = cms.untracked.PSet(
# 	#   limit = cms.untracked.int32(-1)
# 	#),
# 	#TrackAssociator = cms.untracked.PSet(
# 	#   limit = cms.untracked.int32(-1)
# 	#),
#     ),
#     debugModules = cms.untracked.vstring("muons1stStes")
# )

# import FWCore.Framework.test.cmsExceptionsFatalOption_cff
# options = cms.untracked.PSet(
#     Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
# )
process.options = cms.untracked.PSet(
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/205066EB-531A-E711-A026-0025905B8560.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/2CBB627A-5B1A-E711-8044-0CC47A78A496.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/32AEBC8E-581A-E711-8D0D-0CC47A4D76A2.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/3E4ECA76-571A-E711-B439-0CC47A7452D0.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/56CEFCBB-4C1A-E711-A33D-0CC47A4C8ED8.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/760926DD-5B1A-E711-BD0F-0025905B85FE.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/78958AA6-771A-E711-A294-0CC47A4D75F6.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/88A132C9-541A-E711-B0DE-0CC47A4C8EE8.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/944E62B3-771A-E711-982C-0025905B8610.root',
        # '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20_PU50-v1/00000/F6624C6E-4D1A-E711-BA71-0CC47A4D7690.root'
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/18F7C873-5C19-E711-8B21-0CC47A4D7670.root',
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/1A8BB1E0-9D19-E711-9585-0CC47A7C34A0.root',
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/5694119A-5819-E711-914D-0CC47A4D7616.root',
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/68BC6F0D-6B19-E711-B7D5-0CC47A4D768C.root',
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/788BE9DD-6419-E711-B1FB-0CC47A74527A.root',
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/86DE5D32-9F19-E711-8456-0CC47A4D7692.root',
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/DCE77919-5119-E711-8F4F-0CC47A78A468.root',
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/F2FB6238-9C19-E711-8C04-0CC47A7C34A0.root',
        '/store/relval/CMSSW_9_1_0_pre2/RelValZMM_13/GEN-SIM-RECO/PU25ns_90X_upgrade2017_realistic_v20-v1/00000/F49D1692-6819-E711-A05B-0CC47A4D7668.root',
        ),
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/eos/cms/store/user/dmytro/tmp/RelValZMM_13+CMSSW_9_1_0_pre2-PU25ns_90X_upgrade2017_realistic_v20-v1+muonid_3sigma_arbitrated.root'),
    outputCommands = cms.untracked.vstring("drop *",
        "keep *_genParticleCandidates_*_*",
        "keep recoGenParticles_genParticles_*_*",
	"keep recoTracks_*_*_*",
	"keep recoTrackExtras_*_*_*",
	"keep recoMuons_*_*_*",
	"keep *_cscSegments_*_*",
	"keep *_dt4DSegments_*_*",
	"keep *_towerMaker_*_*",
	"keep *_*_*_TEST")
)

# Setup link re-builder
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

process.muonid_study = cms.Path(process.globalMuonLinks* process.muonIdProducerSequence)


# process.muons.fillIsolation = False
# process.muons.minPt = 0.
# process.muons.minP = 0.

process.endjob_step = cms.EndPath(process.endOfProcess)
process.e = cms.EndPath(process.out)

# Schedule definition
process.schedule = cms.Schedule(process.muonid_study,process.endjob_step,process.e)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(8)
process.options.numberOfStreams=cms.untracked.uint32(0)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
