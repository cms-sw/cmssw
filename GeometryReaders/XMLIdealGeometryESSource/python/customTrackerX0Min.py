import FWCore.ParameterSet.Config as cms
def customiseMaterialBudget(process):
    process.XMLFromDBSource.label='ExtendedX0Min'
    return (process)
# foo bar baz
