import FWCore.ParameterSet.Config as cms

process = cms.Process("dqmFeeder")

process.load("FWCore.MessageService.MessageLogger_cfi")
# suppress printout of error messages on every event when a collection is missing in the event
process.MessageLogger.categories.append("EmDQMInvalidRefs")
process.MessageLogger.cerr.EmDQMInvalidRefs = cms.untracked.PSet(limit = cms.untracked.int32(5))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'/store/relval/CMSSW_4_4_0_pre2/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V4-v1/0126/D0D7EB8D-8CA1-E011-910D-0018F3D0960E.root'
        '/store/relval/CMSSW_4_4_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V4-v1/0128/7A69572D-A0A1-E011-8B9B-001BFCDBD19E.root'
        #'/store/relval/CMSSW_4_4_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V4-v1/0125/7EFB2404-7AA1-E011-A05F-00304867908C.root'
        #'/store/relval/CMSSW_4_4_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START43_V4-v1/0120/EC41EA85-39A0-E011-AA76-00304867D836.root'
    )
)

process.load("HLTriggerOffline.Egamma.EgammaValidationAutoConf_cff")

# set output to verbose = all
process.dqmFeeder.verbosity = cms.untracked.uint32(3)
# switch to select between only MC matched histograms or all histograms
#process.dqmFeeder.mcMatchedOnly = cms.untracked.bool(False)
# switch for phi plots
process.dqmFeeder.noPhiPlots = cms.untracked.bool(False)
# switch for 2D isolation plots
process.dqmFeeder.noIsolationPlots = cms.untracked.bool(False)

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
# End of original testEmDQMFeeder_cfg.py
#----------------------------------------
