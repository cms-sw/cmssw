import FWCore.ParameterSet.Config as cms

process = cms.Process("dqmFeeder")

process.load("FWCore.MessageService.MessageLogger_cfi")
# suppress printout of error messages on every event when a collection is missing in the event
process.MessageLogger.categories.append("EmDQMInvalidRefs")
process.MessageLogger.cerr.EmDQMInvalidRefs = cms.untracked.PSet(limit = cms.untracked.int32(5))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'/store/relval/CMSSW_4_2_0_pre7/RelValWE/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0033/BAA28B05-224F-E011-92F7-0026189438D3.root'
        '/store/relval/CMSSW_4_2_0_pre7/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/DA82324C-BC4E-E011-9065-00261894383F.root'
        #'/store/relval/CMSSW_4_2_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START42_V6-v2/0032/4C8D35D6-B24E-E011-9151-003048679030.root'
    )
)

process.dqmFeeder = cms.EDAnalyzer('EmDQMFeeder',
                              processname = cms.string("HLT"),
                              triggerobject = cms.InputTag("hltTriggerSummaryRAW","","HLT"),
                              genEtaAcc = cms.double(2.5),
                              genEtAcc = cms.double(2.0),
                              PtMax = cms.untracked.double(100.0),
                              isData = cms.bool(False)
                             )

process.load("HLTriggerOffline.Egamma.EgammaValidation_cff")

process.p = cms.Path(
                     # require generated particles in fiducial volume
                     process.egammaSelectors *     
                     process.dqmFeeder
                    )

#----------------------------------------
process.post=cms.EDAnalyzer("EmDQMPostProcessor",
                            subDir = cms.untracked.string("HLT/HLTEgammaValidation"),
                            dataSet = cms.untracked.string("unknown"),
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
