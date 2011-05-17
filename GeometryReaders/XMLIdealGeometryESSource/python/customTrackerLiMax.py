import FWCore.ParameterSet.Config as cms
def customise(process):
    process.XMLFromDBSource.label='ExtendedLiMax'
    return (process)
