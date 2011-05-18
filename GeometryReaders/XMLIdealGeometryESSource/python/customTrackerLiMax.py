import FWCore.ParameterSet.Config as cms
def customiseMaterialBudget(process):
    process.XMLFromDBSource.label='ExtendedLiMax'
    return (process)
