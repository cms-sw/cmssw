import FWCore.ParameterSet.Config as cms

process = cms.Process("hVALIDATION")
#Geometry
process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")

#process.load("HLTriggerOffline.Higgs.HLTHiggsBits-WW_cff")
#process.load("HLTriggerOffline.Higgs.HLTHiggsBits-gg_cff")
#process.load("HLTriggerOffline.Higgs.HLTHiggsBits-ZZ_cff")
process.load("HLTriggerOffline.Higgs.HLTHiggsBits-2tau_cff")
#process.load("HLTriggerOffline.Higgs.HLTHiggsBits-taunu_cff")

process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.dqmSaverMy = cms.EDAnalyzer("DQMFileSaver",
        convention=cms.untracked.string("Offline"),
    
          workflow=cms.untracked.string("/HLT/Higgs/Validation"),
        
         dirName=cms.untracked.string("."),
         saveAtJobEnd=cms.untracked.bool(True),                        
         forceRunNumber=cms.untracked.int32(999871)
	)

process.p = cms.Path(process.hltHiggsValidation
	*process.dqmSaverMy
	)


#process.p1 = cms.Path(process.hltHiggsValidation)


