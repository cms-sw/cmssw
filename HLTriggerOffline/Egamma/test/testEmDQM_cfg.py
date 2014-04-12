import FWCore.ParameterSet.Config as cms

process = cms.Process("emdqm")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START70_V1::All'

process.load("FWCore.MessageService.MessageLogger_cfi")
# suppress printout of error messages on every event when a collection is missing in the event
process.MessageLogger.categories.append("EmDQMInvalidRefs")
process.MessageLogger.cerr.EmDQMInvalidRefs = cms.untracked.PSet(limit = cms.untracked.int32(5))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_0_0_pre8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_START70_V1-v1/00000/1C634144-E94A-E311-964E-002618943866.root',
#        '/store/relval/CMSSW_7_0_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/PU_START70_V1-v1/00000/269313CD-E54A-E311-9EB4-002618943894.root',
#        '/store/relval/CMSSW_7_0_0_pre8/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2_RR-v7/00000/96E6CAB6-3B59-E311-9593-002618943925.root',
#        '/store/relval/CMSSW_7_0_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START70_V2_RR-v7/00000/B6FDCDBB-3D59-E311-AB2B-002618943857.root',
    )
)

process.load("HLTriggerOffline.Egamma.EgammaValidationAutoConf_cff")

# set output to verbose = all
process.emdqm.verbosity = cms.untracked.uint32(3)
# switch to select between only MC matched histograms or all histograms
process.emdqm.mcMatchedOnly = cms.untracked.bool(False)
# switch for phi plots
process.emdqm.noPhiPlots = cms.untracked.bool(False)
# switch for 2D isolation plots
process.emdqm.noIsolationPlots = cms.untracked.bool(False)

process.p = cms.Path(
                     # require generated particles in fiducial volume
                     process.egammaSelectors *     
                     process.egammaValidationSequence
                    )

#----------------------------------------
process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation"),
                            dataSet = cms.untracked.string("unknown"),
                            noPhiPlots = cms.untracked.bool(False),
                           )

#process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

#----------------------------------------
# DQM service
#----------------------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

process.ppost = cms.EndPath(process.post+process.dqmSaver)

#----------------------------------------
# End of original testEmDQM_cfg.py
#----------------------------------------
