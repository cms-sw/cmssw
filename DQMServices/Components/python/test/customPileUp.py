
import FWCore.ParameterSet.Config as cms

def customise(process):

    process.mix.input.fileNames = [ '/store/relval/CMSSW_3_9_0_pre2/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0015/24FDE109-6EA8-DF11-82DE-002618943807.root' ]

#    process.load("DQMServices.Components.DQMStoreStats_cfi")
#    process.stats = cms.Path(process.dqmStoreStats)
#    process.schedule.insert(-2,process.stats)

    return(process)

