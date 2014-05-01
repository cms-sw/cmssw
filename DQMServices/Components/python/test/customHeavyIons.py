
import FWCore.ParameterSet.Config as cms

def customise(process):

    process.source.fileNames = [ '/store/hidata/HIRun2010/HIAllPhysics/RAW/v1/000/152/698/44E64367-D9FB-DF11-B46F-001D09F28D54.root']

#    process.load("DQMServices.Components.DQMStoreStats_cfi")
#    process.stats = cms.Path(process.dqmStoreStats)
#    process.schedule.insert(-2,process.stats)

    return(process)

