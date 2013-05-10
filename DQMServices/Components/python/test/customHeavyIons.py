
import FWCore.ParameterSet.Config as cms

def customise(process):

    process.source.fileNames = [ '/store/hidata/HIRun2010/HIAllPhysics/RAW/v1/000/150/887/FC4C9779-03EE-DF11-AAC6-001617C3B778.root']

#    process.load("DQMServices.Components.DQMStoreStats_cfi")
#    process.stats = cms.Path(process.dqmStoreStats)
#    process.schedule.insert(-2,process.stats)

    return(process)

