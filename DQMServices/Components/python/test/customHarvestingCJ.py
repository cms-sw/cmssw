
import FWCore.ParameterSet.Config as cms

def customise(process):
    process.options = cms.untracked.PSet(
        wantSummary = cms.untracked.bool(True) 
        )
    
    process.load("DQMServices.Components.DQMStoreStats_cfi")
    process.stats = cms.EndPath(process.dqmStoreStats)
    
    process.schedule.remove(process.edmtome_step)
    process.schedule.append(process.stats)
    
    return(process)

