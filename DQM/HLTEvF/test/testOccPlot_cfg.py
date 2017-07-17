import FWCore.ParameterSet.Config as cms

process = cms.Process("DQMFourVectorTest")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.load("DQM.HLTEvF.FourVectorHLTOnline_cfi")
#process.load("DQM.HLTEvF.TrigResRateMon_cfi")

process.load("DQM.HLTEvF.OccupancyPlotter_cfi")


process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('file:/data/ndpc0/b/slaunwhj/rawData/June/0091D91D-D19B-E011-BDCE-001D09F2512C.root')
                            fileNames = cms.untracked.vstring('file:/data/ndpc0/c/abrinke1/RAW/170354/SingleMu/08B0697A-B7B0-E011-B0DE-003048D375AA.root')
    #fileNames = cms.untracked.vstring('file:/data/ndpc0/b/slaunwhj/ONLINE/CMSSW_4_2_4_hltpatch1/src/UserCode/slaunwhj/hltTest/outputHLTDQMResults.root')
    #fileNames = cms.untracked.vstring('file:/data/ndpc0/b/slaunwhj/ONLINE/CMSSW_4_2_4_hltpatch1/src/HLTrigger/Configuration/test/outputHLTDQMResults_nAll_newContents.root')
    #fileNames = cms.untracked.vstring('file:/data/ndpc0/b/slaunwhj/ONLINE/CMSSW_4_2_4_hltpatch1/src/HLTrigger/Configuration/test/outputHLTDQMResults_nAll_cosmics_newContents.root')
    # Students made these files                        
    #fileNames = cms.untracked.vstring('file:/data/ndpc0/c/lwming/CMSSW_4_2_4_hltpatch1/src/HLTrigger/Configuration/test/outputHLTDQMResults_r167102_SingleMu.root')
    #fileNames = cms.untracked.vstring('file:/data/ndpc0/c/lwming/CMSSW_4_2_4_hltpatch1/src/HLTrigger/Configuration/test/outputHLTDQMResults_r167281_SingleMu.root')                            
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

################################
#
# Need to do raw to digi 
# in order to use PS providers
#
# This is a hassle
# but I will try it
# following lines are only for
# running the silly RawToDigi
#
################################
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR_R_42_V14::All'
#process.load('Configuration/StandardSequences/RawToDigi_Data_cff')

#process.rateMon  = cms.EndPath(process.RawToDigi*process.trRateMon* process.hltResultsOn)
#process.rateMon  = cms.EndPath(process.trRateMon)

process.pathForOccupancy = cms.EndPath(process.onlineOccPlot)
process.pp = cms.Path(process.dqmEnv+process.dqmSaver)

process.DQMStore.verbose = 0
process.dqmSaver.dirName = '.'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

