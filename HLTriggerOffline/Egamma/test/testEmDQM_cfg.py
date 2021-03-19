import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

process = cms.Process("emdqm")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START72_V1::All'

process.load("FWCore.MessageService.MessageLogger_cfi")
# suppress printout of error messages on every event when a collection is missing in the event
process.MessageLogger.cerr.EmDQMInvalidRefs = cms.untracked.PSet(limit = cms.untracked.int32(5))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:../../../HLTrigger/Configuration/test/outputAForPP.root',
#        '/store/relval/CMSSW_7_1_0_pre7/RelValH130GGgluonfusion_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS171_V7-v1/00000/C87CDC3A-B1D0-E311-8890-02163E00E6DE.root',
#        '/store/relval/CMSSW_7_1_0_pre7/RelValWE_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS171_V7-v1/00000/665BE840-B4D0-E311-BBA6-02163E00E694.root',
#        '/store/relval/CMSSW_7_1_0_pre7/RelValPhotonJets_Pt_10_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS171_V7-v1/00000/C0AB31B9-A2D0-E311-A15D-02163E00E725.root',
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
# which trigger object and process should we run on?
#process.emdqm.triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLTTEST")

process.p = cms.Path(
                     # require generated particles in fiducial volume
                     process.egammaSelectors *     
                     process.egammaValidationSequence
                    )

#----------------------------------------
process.post=DQMEDHarvester("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation"),
                            dataSet = cms.untracked.string("unknown"),
                            noPhiPlots = cms.untracked.bool(False),
                            ignoreEmpty = cms.untracked.bool(False),
                           )

#process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

#----------------------------------------
# DQM service
#----------------------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/RelVal/HLTriggerOffline/Egamma'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)

process.ppost = cms.EndPath(process.post+process.dqmSaver)

#----------------------------------------
# End of original testEmDQM_cfg.py
#----------------------------------------
