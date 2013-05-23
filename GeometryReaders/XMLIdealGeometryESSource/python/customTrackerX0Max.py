import FWCore.ParameterSet.Config as cms
def customiseMaterialBudget(process):
    process.XMLFromDBSource.label='ExtendedX0Max'
    return (process)
