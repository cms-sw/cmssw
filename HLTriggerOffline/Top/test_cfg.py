import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load("HLTriggerOffline.Top.topvalidation_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")




process.source = cms.Source("PoolSource",
    #AlCaReco File
    fileNames = cms.untracked.vstring(
     '/store/relval/CMSSW_2_1_8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0002/04983078-9082-DD11-BB8C-0019DB2F3F9B.root',  
     '/store/relval/CMSSW_2_1_8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0003/F8FA9F38-9182-DD11-AE18-001617C3B76E.root',
     '/store/relval/CMSSW_2_1_8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0003/FC224B80-9082-DD11-B1DE-000423D94E70.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


#process.FEVT = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
#    fileName = cms.untracked.string('reco-gr-dqm.root')
#)

#process.p = cms.Path(process.muonAlignment*process.MEtoEDMConverter)

process.load("DQMServices.Core.DQM_cfg")

process.dqmSaverMy = cms.EDFilter("DQMFileSaver",
        convention=cms.untracked.string("Offline"),
    
          workflow=cms.untracked.string("/HLT/Top/Validation"),
        
         dirName=cms.untracked.string("."),
         saveAtJobEnd=cms.untracked.bool(True),                        
         forceRunNumber=cms.untracked.int32(999871)
	)

process.p = cms.Path(process.HLTTopVal
	*process.dqmSaverMy
	)





