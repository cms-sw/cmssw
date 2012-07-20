# t0est file for PFDQM Skimming
# it reads events and run runnumbers from a DQM output file and creates
# a skimmed output for bad events
process = cms.Process('PFlowSkim')
#------------------------
# Message Logger Settings
#------------------------
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
#--------------------------------------
# Event Source & # of Events to process
#---------------------------------------
process.source = cms.Source("PoolSource",
                   fileNames = cms.untracked.vstring()
                 )
process.maxEvents = cms.untracked.PSet(
                      input = cms.untracked.int32(-1)
                    )
process.load("DQMServices.Core.DQM_cfg")

#--------------
# PFlow Skim 
#--------------
process.load("DQMOffline.PFTau.PFDQMEventSelector_cfi")
process.pfDQMEventSelector.InputFileName = "DQM_V0001_R000000001__PFlow__Validation__QCD.root"
process.pfDQMEventSelector.DebugOn = True
process.pfDQMEventSelector.FolderNames = cms.vstring("PFJetValidation/CompWithGenJet")

process.eventSelectorPath = cms.Path(process.pfDQMEventSelector)

process.outputSkim = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('eventSelectorPath')),
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RAW-RECO'),
                                         filterName = cms.untracked.string('PFDQMEventSelector')),
                               fileName = cms.untracked.string('./test.root')
                               )
process.outpath = cms.EndPath(process.outputSkim)


#--------------------------------------
# List File from where events will be skimmed
#---------------------------------------
process.PoolSource.fileNames = [
    '/store/relval/CMSSW_4_2_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_42_V7-v1/0042/2CC0574D-9556-E011-A92B-0018F3D09658.root'
]
