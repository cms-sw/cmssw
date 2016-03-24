import FWCore.ParameterSet.Config as cms

source = cms.Source("TotemStandaloneRawDataSource",
    # if non-zero, prints a file summary in the beginning
    verbosity = cms.untracked.uint32(1),
    
    # event number will be printed every 'printProgressFrequency' events,
    # nothing printed if 0
    printProgressFrequency = cms.untracked.uint32(0),

    # the list of files to be processed
    fileNames = cms.untracked.vstring()
)
